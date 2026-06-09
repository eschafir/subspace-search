"""
Phase 0 — Reproduce vanilla RandOpt on GSM8K.

Timeline target: 1 week (§7 of proposal).
Expected outcome: RandOpt (N=500, K=50) improves over baseline on Qwen2.5-1.5B.

Usage:
    python experiments/phase0_reproduce.py
    python experiments/phase0_reproduce.py --model qwen-3b --N 500 --K 50
    python experiments/phase0_reproduce.py --score-mode loss  # faster, for debugging
"""
import argparse
import json
import os
import sys
import time
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.models import load, with_delta, param_count, best_gpu
from src.perturbation import make_isotropic_delta
from src.benchmarks import gsm8k
from src.evaluate import score_examples_generation, score_examples_loss
from src.randopt import randopt


def build_score_fn(mode: str, tokenizer, max_new_tokens: int, batch_size: int):
    """Return a score_fn(model, tokenizer, examples, device) -> float."""
    if mode == "generation":
        def score_fn(model, tok, examples, device):
            return score_examples_generation(
                model, tok, examples, device,
                format_fn=gsm8k.format_prompt,
                correct_fn=lambda text, ex: gsm8k.is_correct(
                    gsm8k.extract_answer(text),
                    gsm8k.get_reference_answer(ex),
                ),
                max_new_tokens=max_new_tokens,
                batch_size=batch_size,
            )
    elif mode == "loss":
        def score_fn(model, tok, examples, device):
            return score_examples_loss(
                model, tok, examples, device,
                format_full_fn=gsm8k.format_full,
                format_prompt_fn=gsm8k.format_prompt,
                batch_size=batch_size,
            )
    else:
        raise ValueError(f"Unknown score mode: {mode}")
    return score_fn


def majority_vote_test(model, tokenizer, d_test, top_seeds, top_sigmas,
                       device, max_new_tokens, batch_size):
    """
    Run majority vote over top-K perturbed models on the test set.

    Outer loop: seeds (apply perturbation once, then batch all test examples).
    Inner loop: batched generation over d_test.
    Cost: K × ceil(|d_test| / batch_size) batched generation calls.
    """
    from collections import Counter
    from tqdm import tqdm

    d = param_count(model)
    # all_answers[i] collects one answer per seed for test example i
    all_answers = [[] for _ in range(len(d_test))]

    for seed, sigma in tqdm(zip(top_seeds, top_sigmas), total=len(top_seeds),
                            desc="Majority vote (seeds)"):
        delta = make_isotropic_delta(seed, d, sigma)
        with with_delta(model, delta):
            for batch_start in range(0, len(d_test), batch_size):
                batch = d_test[batch_start: batch_start + batch_size]
                prompts = [gsm8k.format_prompt(ex, tokenizer) for ex in batch]
                inputs = tokenizer(
                    prompts, return_tensors="pt", padding=True,
                    truncation=True, max_length=1024,
                ).to(device)
                prompt_len = inputs["input_ids"].shape[1]

                with torch.no_grad():
                    out = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id,
                    )

                for i, ex_idx in enumerate(range(batch_start,
                                                  batch_start + len(batch))):
                    text = tokenizer.decode(out[i, prompt_len:],
                                            skip_special_tokens=True)
                    all_answers[ex_idx].append(gsm8k.extract_answer(text))

    n_correct = 0
    for i, ex in enumerate(d_test):
        voted = Counter(a for a in all_answers[i] if a is not None)
        best  = voted.most_common(1)[0][0] if voted else None
        if gsm8k.is_correct(best, gsm8k.get_reference_answer(ex)):
            n_correct += 1

    return n_correct / len(d_test)


def main(args):
    device = best_gpu()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    print(f"Loading {args.model} on {device}...")
    model, tokenizer = load(args.model, device=device)
    print(f"  Parameters: {param_count(model):,}")

    d_train, d_test = gsm8k.load_gsm8k(
        n_train=args.n_train, n_test=args.n_test, seed=args.seed
    )
    print(f"  D_train={len(d_train)}, D_test={len(d_test)}")

    checkpoint_path = args.output.replace(".json", "_checkpoint.json")

    if args.resume_from and os.path.exists(args.resume_from):
        print(f"\nResuming from {args.resume_from}...")
        with open(args.resume_from) as f:
            ckpt = json.load(f)
        baseline_acc = ckpt["baseline_acc"]
        selection    = {k: ckpt[k] for k in
                        ("top_seeds", "top_sigmas", "top_scores", "all_scores",
                         "all_seeds", "all_sigmas")}
        print(f"  Loaded baseline_acc={baseline_acc:.4f}, "
              f"top-1 train score={selection['top_scores'][0]:.4f}")
    else:
        score_fn = build_score_fn(args.score_mode, tokenizer, args.max_new_tokens, args.batch_size)

        # Baseline (no perturbation)
        print("\nBaseline accuracy...")
        t0 = time.time()
        baseline_acc = score_examples_generation(
            model, tokenizer, d_test, device,
            format_fn=gsm8k.format_prompt,
            correct_fn=lambda text, ex: gsm8k.is_correct(
                gsm8k.extract_answer(text), gsm8k.get_reference_answer(ex)
            ),
            max_new_tokens=args.max_new_tokens,
            batch_size=args.batch_size,
        )
        print(f"  Baseline: {baseline_acc:.4f}  ({time.time()-t0:.0f}s)")

        # Vanilla RandOpt selection
        print(f"\nRandOpt selection (N={args.N}, K={args.K}, mode={args.score_mode})...")
        t0 = time.time()
        selection = randopt(
            model, tokenizer, d_train,
            N=args.N, K=args.K,
            sigmas=args.sigmas,
            device=device,
            score_fn=score_fn,
            base_seed=args.seed,
        )
        print(f"  Selection done in {time.time()-t0:.0f}s")
        print(f"  Top-1 train score: {selection['top_scores'][0]:.4f}")
        print(f"  Top-{args.K} train scores: min={min(selection['top_scores']):.4f}, "
              f"mean={sum(selection['top_scores'])/len(selection['top_scores']):.4f}")

        # Save checkpoint immediately so majority vote can be resumed independently
        ckpt = {"model": args.model, "baseline_acc": baseline_acc, **selection}
        with open(checkpoint_path, "w") as f:
            json.dump(ckpt, f, indent=2)
        print(f"  Checkpoint saved to {checkpoint_path}")

    # Majority vote test accuracy
    print("\nMajority vote on test set...")
    t0 = time.time()
    mv_acc = majority_vote_test(
        model, tokenizer, d_test,
        selection["top_seeds"], selection["top_sigmas"],
        device, args.max_new_tokens, args.batch_size,
    )
    print(f"  RandOpt (K={args.K}) test accuracy: {mv_acc:.4f}  ({time.time()-t0:.0f}s)")
    print(f"  Improvement over baseline: {mv_acc - baseline_acc:+.4f}")

    results = {
        "model":        args.model,
        "N":            args.N,
        "K":            args.K,
        "score_mode":   args.score_mode,
        "baseline_acc": baseline_acc,
        "mv_acc":       mv_acc,
        "delta":        mv_acc - baseline_acc,
        **selection,
    }

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",         default="qwen-1.5b",
                        choices=["qwen-1.5b", "qwen-3b"])
    parser.add_argument("--N",             type=int,   default=500)
    parser.add_argument("--K",             type=int,   default=50)
    parser.add_argument("--sigmas",        type=float, nargs="+",
                        default=[1e-3, 2e-3, 3e-3])
    parser.add_argument("--n-train",       type=int,   default=200)
    parser.add_argument("--n-test",        type=int,   default=200)
    parser.add_argument("--max-new-tokens",type=int,   default=512)
    parser.add_argument("--batch-size",    type=int,   default=8)
    parser.add_argument("--seed",          type=int,   default=42)
    parser.add_argument("--score-mode",    default="loss",
                        choices=["generation", "loss"],
                        help="'loss' is 10x faster; 'generation' is faithful to paper")
    parser.add_argument("--output",        default="results/phase0.json")
    parser.add_argument("--resume-from",   default=None,
                        help="Path to a _checkpoint.json to skip selection and go straight to majority vote")
    main(parser.parse_args())
