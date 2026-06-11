"""
Parameter-wise Fisher Sparsity RandOpt.

This script implements the axis-aligned coordinate-sparse subspace search:
  1. Computes the empirical Fisher diagonal (average squared gradients) on D_loc.
  2. Masks out all but the top-M most sensitive parameters.
  3. Samples random weight perturbations only inside this sparse coordinate subspace.
  4. Runs RandOpt selection on D_train and majority vote on D_test.

Optimizations:
  * Sparse memory context manager: only transfers and updates the top-M parameters (100k)
    in-place on GPU, bypassing the overhead of transferring 1.5B parameters on every step.
  * Checkpoint resuming: saves selection state and top_indices, allowing immediate resuming
    and bypassing the selection phase if it was already completed.
"""
import argparse
import json
import os
import sys
import time
import contextlib
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.models import load, with_delta, param_count, best_gpu, trainable_params
from src.benchmarks import gsm8k
from src.evaluate import score_examples_loss, score_examples_generation


def compute_fisher_diagonal(
    model,
    tokenizer,
    examples: list[dict],
    device: str,
    format_full_fn,
    format_prompt_fn,
) -> torch.Tensor:
    """Compute empirical Fisher diagonal (mean squared SFT gradients) on CPU."""
    model.train()
    d = param_count(model)
    fisher_diagonal = torch.zeros(d, dtype=torch.float32, device="cpu")
    
    for ex in tqdm(examples, desc="Fisher Diagonal (per-sample grad)"):
        model.zero_grad()
        
        full_text   = format_full_fn(ex, tokenizer)
        prompt_text = format_prompt_fn(ex, tokenizer)

        full_enc   = tokenizer(full_text,   return_tensors="pt",
                                truncation=True, max_length=1024).to(device)
        prompt_len = tokenizer(prompt_text, return_tensors="pt",
                                truncation=True, max_length=1024
                                )["input_ids"].shape[1]

        with torch.enable_grad():
            out = model(**full_enc, use_cache=False)
            logits = out.logits[0]          # (seq, vocab)
            labels = full_enc["input_ids"][0].clone()
            labels[:prompt_len] = -100      # mask prompt tokens

            shift_logits = logits[:-1]
            shift_labels = labels[1:]

            loss = F.cross_entropy(shift_logits, shift_labels,
                                   ignore_index=-100, reduction="mean")
            loss.backward()

        # Collect gradients and add squared values to CPU tensor
        grads = []
        for p in model.parameters():
            if p.requires_grad:
                g = (p.grad.detach().float() if p.grad is not None
                     else torch.zeros(p.numel(), dtype=torch.float32, device=device))
                grads.append(g.flatten().cpu())
        
        flat_grad = torch.cat(grads)
        fisher_diagonal.add_(flat_grad.pow(2))
        
    model.zero_grad()
    model.eval()
    
    return fisher_diagonal / len(examples)


@contextlib.contextmanager
def with_sparse_delta(param_mappings, z: torch.Tensor):
    """
    Temporarily apply a sparse delta vector z (M elements) to model parameters.
    Only transfers and updates the active coordinates in-place on GPU.
    """
    for p, local_idxs, matching_global_idxs in param_mappings:
        chunk = z[matching_global_idxs].to(dtype=p.dtype, device=p.device)
        p.data.view(-1)[local_idxs].add_(chunk)
    try:
        yield
    finally:
        for p, local_idxs, matching_global_idxs in param_mappings:
            chunk = z[matching_global_idxs].to(dtype=p.dtype, device=p.device)
            p.data.view(-1)[local_idxs].sub_(chunk)


def score_perturbation_sparse(
    model,
    tokenizer,
    examples: list[dict],
    device: str,
    param_mappings,
    z: torch.Tensor,
    score_fn,
) -> float:
    """Score model with temporarily applied sparse delta z."""
    with with_sparse_delta(param_mappings, z):
        return score_fn(model, tokenizer, examples, device)


def majority_vote_test_sparse(model, tokenizer, d_test, top_seeds, param_mappings, M, sigma,
                              device, max_new_tokens, batch_size):
    """Run majority vote test inference using optimized sparse delta application."""
    from collections import Counter
    
    n_correct = 0
    for ex in tqdm(d_test, desc="Majority vote test"):
        prompt  = gsm8k.format_prompt(ex, tokenizer)
        inputs  = tokenizer(prompt, return_tensors="pt",
                            truncation=True, max_length=1024).to(device)
        plen    = inputs["input_ids"].shape[1]

        answers = []
        for seed in top_seeds:
            g = torch.Generator()
            g.manual_seed(seed)
            z = torch.randn(M, generator=g, dtype=torch.float32) * sigma
            
            with with_sparse_delta(param_mappings, z):
                with torch.no_grad():
                    out = model.generate(
                        **inputs, max_new_tokens=max_new_tokens,
                        do_sample=False, pad_token_id=tokenizer.pad_token_id,
                    )
            text = tokenizer.decode(out[0, plen:], skip_special_tokens=True)
            answers.append(gsm8k.extract_answer(text))

        voted = Counter(a for a in answers if a is not None)
        best  = voted.most_common(1)[0][0] if voted else None
        if gsm8k.is_correct(best, gsm8k.get_reference_answer(ex)):
            n_correct += 1

    return n_correct / len(d_test)


def main(args):
    device = best_gpu()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    print(f"Loading {args.model}...")
    model, tokenizer = load(args.model, device=device)
    d = param_count(model)

    d_train, d_test = gsm8k.load_gsm8k(
        n_train=args.n_train, n_test=args.n_test, seed=args.seed
    )
    d_loc = d_train[:args.n_loc]

    # Load baseline accuracy if available
    baseline_acc = None
    if args.phase0_results and os.path.exists(args.phase0_results):
        print(f"Loading baseline accuracy from {args.phase0_results}...")
        with open(args.phase0_results) as f:
            p0 = json.load(f)
        baseline_acc = p0.get("baseline_acc")
        
    if baseline_acc is None:
        print("\nComputing baseline accuracy...")
        baseline_acc = score_examples_generation(
            model, tokenizer, d_test, device,
            format_fn=gsm8k.format_prompt,
            correct_fn=lambda t, ex: gsm8k.is_correct(
                gsm8k.extract_answer(t), gsm8k.get_reference_answer(ex)),
            max_new_tokens=args.max_new_tokens, batch_size=args.batch_size,
        )
    print(f"  Baseline Accuracy: {baseline_acc:.4f}")

    # Checkpoint setup
    ckpt_path = args.output.replace(".json", "_checkpoint.json")
    all_scores = []
    all_seeds = []
    top_indices = None
    start_i = 0

    if os.path.exists(ckpt_path):
        print(f"\nLoading selection checkpoint from {ckpt_path}...")
        with open(ckpt_path) as f:
            ckpt = json.load(f)
        all_scores = ckpt["all_scores"]
        all_seeds = ckpt["all_seeds"]
        top_indices = torch.tensor(ckpt["top_indices"])
        start_i = ckpt["last_index"] + 1
        print(f"  Loaded checkpoint: {len(all_scores)} completed iterations. Top indices shape: {top_indices.shape}")
    else:
        # Compute Fisher diagonal
        print(f"\nComputing Fisher diagonal on D_loc ({len(d_loc)})...")
        model.gradient_checkpointing_enable()
        fisher_diagonal = compute_fisher_diagonal(
            model, tokenizer, d_loc, device,
            format_full_fn=gsm8k.format_full,
            format_prompt_fn=gsm8k.format_prompt,
        )
        model.gradient_checkpointing_disable()

        # Get top-M parameter indices
        print(f"Selecting top-M={args.M} sensitive parameters...")
        fisher_gpu = fisher_diagonal.to(device)
        _, top_indices = torch.topk(fisher_gpu, args.M, largest=True)
        top_indices = top_indices.cpu()
        del fisher_gpu
        torch.cuda.empty_cache()

    # Precompute parameter mapping for optimized sparse updates
    print("Precomputing layer-wise mappings for sparse VRAM updates...")
    params = trainable_params(model)
    offset = 0
    param_mappings = []
    for p in params:
        n = p.numel()
        mask = (top_indices >= offset) & (top_indices < offset + n)
        if mask.any():
            matching_global_idxs = torch.where(mask)[0]
            local_idxs = top_indices[matching_global_idxs] - offset
            param_mappings.append((p, local_idxs, matching_global_idxs))
        offset += n
    print(f"  Sparsity mapped successfully. {len(param_mappings)} parameter tensors affected.")

    # Define score function (teacher-forced SFT loss)
    def score_fn(model, tok, examples, device):
        return score_examples_loss(
            model, tok, examples, device,
            format_full_fn=gsm8k.format_full,
            format_prompt_fn=gsm8k.format_prompt,
            batch_size=args.batch_size,
        )

    # Sparsity RandOpt Selection
    if start_i < args.N:
        print(f"\nSparsity RandOpt selection (N={args.N}, K={args.K}, M={args.M})...")
        t0 = time.time()
        
        for i in tqdm(range(start_i, args.N), desc="Sparsity RandOpt sampling"):
            seed = args.seed + i
            g = torch.Generator()
            g.manual_seed(seed)
            z = torch.randn(args.M, generator=g, dtype=torch.float32) * args.sigma
            
            score = score_perturbation_sparse(model, tokenizer, d_train, device, param_mappings, z, score_fn)
            all_scores.append(score)
            all_seeds.append(seed)
            
            # Save checkpoint every 50 iterations
            if (i + 1) % 50 == 0 or (i + 1) == args.N:
                with open(ckpt_path, "w") as f:
                    json.dump({
                        "last_index": i,
                        "all_scores": all_scores,
                        "all_seeds": all_seeds,
                        "top_indices": top_indices.tolist()
                    }, f, indent=2)

        print(f"  Selection completed/resumed in {time.time()-t0:.0f}s")
        
    ranked = sorted(zip(all_scores, all_seeds), reverse=True)
    top_k  = ranked[:args.K]
    top_seeds = [s for _, s in top_k]
    
    print(f"  Top-1 SFT Loss: {top_k[0][0]:.4f}")
    
    # Majority vote test accuracy
    print("\nRunning majority vote on test set...")
    t0 = time.time()
    mv_acc = majority_vote_test_sparse(
        model, tokenizer, d_test, top_seeds, param_mappings, args.M, args.sigma,
        device, args.max_new_tokens, args.batch_size,
    )
    print(f"  Sparsity RandOpt (K={args.K}) test accuracy: {mv_acc:.4f}  ({time.time()-t0:.0f}s)")
    print(f"  Improvement over baseline: {mv_acc - baseline_acc:+.4f}")

    # Save final results
    output = {
        "model": args.model,
        "N": args.N,
        "K": args.K,
        "M": args.M,
        "sigma": args.sigma,
        "baseline_acc": baseline_acc,
        "mv_acc": mv_acc,
        "delta": mv_acc - baseline_acc,
        "top_seeds": top_seeds,
    }
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved results to {args.output}")

    # Remove temporary selection checkpoint on success
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",          default="qwen-1.5b")
    parser.add_argument("--N",              type=int,   default=500)
    parser.add_argument("--K",              type=int,   default=50)
    parser.add_argument("--M",              type=int,   default=100000,
                        help="Number of active parameters in sparse subspace")
    parser.add_argument("--sigma",          type=float, default=2e-3)
    parser.add_argument("--n-train",        type=int,   default=200)
    parser.add_argument("--n-test",         type=int,   default=200)
    parser.add_argument("--n-loc",          type=int,   default=50)
    parser.add_argument("--max-new-tokens", type=int,   default=512)
    parser.add_argument("--batch-size",     type=int,   default=8)
    parser.add_argument("--seed",           type=int,   default=42)
    parser.add_argument("--phase0-results", default="results/phase0_checkpoint.json")
    parser.add_argument("--output",         default="results/sparsity_results.json")
    main(parser.parse_args())
