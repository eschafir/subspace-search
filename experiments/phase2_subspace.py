"""
Phase 2 — Subspace RandOpt (Algorithm 2 from §3.2.4).

The critical comparison: Subspace RandOpt (N=500, S_r) vs. RandOpt (N=500, R^d).
Same compute budget, different geometry. Any accuracy gain is attributable to
restricting search to the gradient subspace.

Usage:
    python experiments/phase2_subspace.py
    python experiments/phase2_subspace.py --rank 100 --N 500 --K 50
"""
import argparse
import json
import os
import sys
import time
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.models import load, with_delta, param_count
from src.perturbation import make_isotropic_delta, make_subspace_delta
from src.benchmarks import gsm8k
from src.evaluate import score_examples_loss, score_examples_generation
from src.randopt import randopt, subspace_randopt
from src.subspace import compute_gradient, build_subspace
from src.metrics import spectral_discordance


def majority_vote_test(model, tokenizer, d_test, top_seeds, top_sigmas,
                       delta_fn, device, max_new_tokens, batch_size):
    """Run majority vote inference for a set of (seed, sigma) pairs."""
    from collections import Counter
    from tqdm import tqdm

    n_correct = 0
    for ex in tqdm(d_test, desc="Majority vote"):
        prompt  = gsm8k.format_prompt(ex["question"], tokenizer)
        inputs  = tokenizer(prompt, return_tensors="pt",
                            truncation=True, max_length=1024).to(device)
        plen    = inputs["input_ids"].shape[1]

        answers = []
        for seed, sigma in zip(top_seeds, top_sigmas):
            delta = delta_fn(seed, sigma)
            with with_delta(model, delta):
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


def compute_score_matrix(model, tokenizer, d_test, all_seeds, all_sigmas,
                         delta_fn, device, max_new_tokens=64, n_sample=50):
    """
    Build score matrix P ∈ R^{N×1} for spectral discordance.
    (Extend to multiple tasks for full discordance measurement.)
    Currently returns per-seed accuracy on d_test (single task).
    """
    from tqdm import tqdm

    scores = []
    for seed, sigma in tqdm(zip(all_seeds[:n_sample], all_sigmas[:n_sample]),
                            total=n_sample, desc="Score matrix"):
        delta = delta_fn(seed, sigma)
        with with_delta(model, delta):
            acc = score_examples_generation(
                model, tokenizer, d_test[:50], device,
                format_fn=gsm8k.format_prompt,
                correct_fn=lambda t, ex: gsm8k.is_correct(
                    gsm8k.extract_answer(t), gsm8k.get_reference_answer(ex)),
                max_new_tokens=max_new_tokens,
                batch_size=8,
            )
        scores.append(acc)
    return scores


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    print(f"Loading {args.model}...")
    model, tokenizer = load(args.model, device=device)
    d = param_count(model)

    d_train, d_test = gsm8k.load_gsm8k(
        n_train=args.n_train, n_test=args.n_test, seed=args.seed
    )
    d_loc = d_train[:args.n_loc]

    def score_fn(model, tok, examples, device):
        return score_examples_loss(
            model, tok, examples, device,
            format_full_fn=gsm8k.format_full,
            format_prompt_fn=gsm8k.format_prompt,
            batch_size=args.batch_size,
        )

    # --- Baseline ---
    print("\nBaseline accuracy...")
    baseline_acc = score_examples_generation(
        model, tokenizer, d_test, device,
        format_fn=gsm8k.format_prompt,
        correct_fn=lambda t, ex: gsm8k.is_correct(
            gsm8k.extract_answer(t), gsm8k.get_reference_answer(ex)),
        max_new_tokens=args.max_new_tokens, batch_size=args.batch_size,
    )
    print(f"  {baseline_acc:.4f}")

    # --- Compute gradient subspace ---
    print(f"\nBuilding gradient subspace (rank={args.rank}, D_loc={len(d_loc)})...")
    model.gradient_checkpointing_enable()
    grad = compute_gradient(
        model, tokenizer, d_loc, device,
        format_full_fn=gsm8k.format_full,
        format_prompt_fn=gsm8k.format_prompt,
    )
    model.gradient_checkpointing_disable()
    U_r, V_r, m, n, pad = build_subspace(grad, rank=args.rank)
    print(f"  Subspace shape: U_r={U_r.shape}, V_r={V_r.shape}")

    # --- Compute-matched RandOpt baseline (N=500, isotropic) ---
    print(f"\nIsotropic RandOpt (N={args.N}, R^d)...")
    t0 = time.time()
    iso_result = randopt(
        model, tokenizer, d_train,
        N=args.N, K=args.K, sigmas=args.sigmas,
        device=device, score_fn=score_fn, base_seed=args.seed,
    )
    iso_acc = majority_vote_test(
        model, tokenizer, d_test,
        iso_result["top_seeds"], iso_result["top_sigmas"],
        lambda s, σ: make_isotropic_delta(s, d, σ),
        device, args.max_new_tokens, args.batch_size,
    )
    print(f"  Isotropic RandOpt (N={args.N}) test acc: {iso_acc:.4f}  ({time.time()-t0:.0f}s)")

    # --- Subspace RandOpt ---
    print(f"\nSubspace RandOpt (N={args.N}, r={args.rank})...")
    t0 = time.time()
    sub_result = subspace_randopt(
        model, tokenizer, d_train,
        N=args.N, K=args.K, sigma=args.sigmas[1],  # middle sigma
        device=device, score_fn=score_fn,
        U_r=U_r, V_r=V_r, m=m, n=n, pad=pad,
        base_seed=args.seed + 10000,  # different seed space from isotropic
    )
    sub_acc = majority_vote_test(
        model, tokenizer, d_test,
        sub_result["top_seeds"], sub_result["top_sigmas"],
        lambda s, σ: make_subspace_delta(s, U_r, V_r, m, n, pad, σ),
        device, args.max_new_tokens, args.batch_size,
    )
    print(f"  Subspace RandOpt (N={args.N}) test acc:  {sub_acc:.4f}  ({time.time()-t0:.0f}s)")

    print(f"\nSummary:")
    print(f"  Baseline:               {baseline_acc:.4f}")
    print(f"  Isotropic N={args.N}:   {iso_acc:.4f}  ({iso_acc-baseline_acc:+.4f})")
    print(f"  Subspace  N={args.N}:   {sub_acc:.4f}  ({sub_acc-baseline_acc:+.4f})")
    print(f"  Subspace vs Isotropic:  {sub_acc-iso_acc:+.4f}")

    output = {
        "model":        args.model,
        "rank":         args.rank,
        "N":            args.N,
        "K":            args.K,
        "baseline_acc": baseline_acc,
        "isotropic_acc": iso_acc,
        "subspace_acc": sub_acc,
        "delta_vs_baseline_iso": iso_acc - baseline_acc,
        "delta_vs_baseline_sub": sub_acc - baseline_acc,
        "delta_subspace_vs_iso": sub_acc - iso_acc,
    }
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",          default="qwen-1.5b")
    parser.add_argument("--rank",           type=int,   default=100)
    parser.add_argument("--N",              type=int,   default=500)
    parser.add_argument("--K",              type=int,   default=50)
    parser.add_argument("--sigmas",         type=float, nargs="+",
                        default=[1e-3, 2e-3, 3e-3])
    parser.add_argument("--n-train",        type=int,   default=200)
    parser.add_argument("--n-test",         type=int,   default=200)
    parser.add_argument("--n-loc",          type=int,   default=50)
    parser.add_argument("--max-new-tokens", type=int,   default=512)
    parser.add_argument("--batch-size",     type=int,   default=8)
    parser.add_argument("--seed",           type=int,   default=42)
    parser.add_argument("--output",         default="results/phase2.json")
    main(parser.parse_args())
