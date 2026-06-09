"""
Phase 1 — Subspace alignment verification (Algorithm 1 from §3.1.2).

Hypothesis 2: top-K RandOpt deltas have ρ̄⁺_r ≥ 2 × ρ̄⁻_r for r ≤ 200.

Steps:
  1. Run vanilla RandOpt (N=500) to identify top-K=50 and K random non-top seeds
  2. Compute task gradient on D_loc (50 examples) and build gradient subspace
  3. Measure ρ_r(Δ) for each delta at ranks r ∈ {10, 50, 100, 200, 500}
  4. Report ρ̄⁺, ρ̄⁻, ratio, and optional Spearman correlation between ρ_r and score

Usage:
    python experiments/phase1_verify.py
    python experiments/phase1_verify.py --phase0-results results/phase0.json
"""
import argparse
import json
import os
import sys
import torch
import numpy as np
from scipy.stats import spearmanr
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.models import load, param_count, best_gpu
from src.perturbation import make_isotropic_delta
from src.benchmarks import gsm8k
from src.evaluate import score_examples_loss
from src.randopt import randopt
from src.subspace import compute_gradient, build_subspace, subspace_energy_ratio
from src.metrics import alignment_ratio


def main(args):
    device = best_gpu()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    print(f"Loading {args.model}...")
    model, tokenizer = load(args.model, device=device)
    d = param_count(model)

    d_train, _ = gsm8k.load_gsm8k(n_train=args.n_train, seed=args.seed)
    # D_loc is a separate small subset for gradient computation
    d_loc = d_train[:args.n_loc]
    print(f"  D_train={len(d_train)}, D_loc={len(d_loc)}")

    # --- Step 1: Run vanilla RandOpt to get top-K and non-top seeds ---
    if args.phase0_results and os.path.exists(args.phase0_results):
        print(f"Loading Phase 0 results from {args.phase0_results}...")
        with open(args.phase0_results) as f:
            p0 = json.load(f)
        top_seeds    = p0["top_seeds"][:args.K]
        top_sigmas   = p0["top_sigmas"][:args.K]
        all_seeds    = p0["all_seeds"]
        all_sigmas   = p0["all_sigmas"]
        all_scores   = p0["all_scores"]
    else:
        print(f"Running RandOpt (N={args.N})...")

        def score_fn(model, tok, examples, device):
            return score_examples_loss(
                model, tok, examples, device,
                format_full_fn=gsm8k.format_full,
                format_prompt_fn=gsm8k.format_prompt,
                batch_size=args.batch_size,
            )

        selection = randopt(
            model, tokenizer, d_train,
            N=args.N, K=args.K,
            sigmas=[1e-3, 2e-3, 3e-3],
            device=device,
            score_fn=score_fn,
            base_seed=args.seed,
        )
        top_seeds  = selection["top_seeds"]
        top_sigmas = selection["top_sigmas"]
        all_seeds  = selection["all_seeds"]
        all_sigmas = selection["all_sigmas"]
        all_scores = selection["all_scores"]

    # Sample K non-top seeds
    top_set = set(top_seeds)
    non_top_idx = [i for i, s in enumerate(all_seeds) if s not in top_set]
    rng = np.random.default_rng(args.seed)
    chosen = rng.choice(non_top_idx, size=min(args.K, len(non_top_idx)), replace=False)
    non_top_seeds  = [all_seeds[i]  for i in chosen]
    non_top_sigmas = [all_sigmas[i] for i in chosen]

    # --- Step 2: Gradient subspace ---
    print(f"\nComputing gradient on D_loc ({len(d_loc)} examples)...")
    model.gradient_checkpointing_enable()
    grad = compute_gradient(
        model, tokenizer, d_loc, device,
        format_full_fn=gsm8k.format_full,
        format_prompt_fn=gsm8k.format_prompt,
    )
    model.gradient_checkpointing_disable()
    print(f"  Gradient norm: {grad.norm():.4f}")

    # --- Step 3: Measure ρ_r at multiple ranks ---
    ranks = args.ranks
    results_by_rank = {}

    for r in ranks:
        print(f"\nBuilding subspace (rank={r})...")
        U_r, V_r, m, n, pad = build_subspace(grad, rank=r)

        rho_plus = [
            subspace_energy_ratio(make_isotropic_delta(s, d, sig), U_r, V_r, m, n, pad)
            for s, sig in tqdm(zip(top_seeds, top_sigmas), total=len(top_seeds),
                               desc=f"  r={r:4d} top-K  ", leave=False)
        ]
        rho_minus = [
            subspace_energy_ratio(make_isotropic_delta(s, d, sig), U_r, V_r, m, n, pad)
            for s, sig in tqdm(zip(non_top_seeds, non_top_sigmas), total=len(non_top_seeds),
                               desc=f"  r={r:4d} non-top", leave=False)
        ]

        mean_plus  = float(np.mean(rho_plus))
        mean_minus = float(np.mean(rho_minus))
        ratio      = alignment_ratio(mean_plus, mean_minus)

        print(f"  r={r:4d}:  ρ̄⁺={mean_plus:.3e}  ρ̄⁻={mean_minus:.3e}  "
              f"ratio={ratio:.2f}x  {'✓ PROCEED' if ratio >= 2 else '✗ weak'}")

        # Spearman correlation between ρ_r(all seeds) and their scores
        all_rho = [
            subspace_energy_ratio(make_isotropic_delta(s, d, sig), U_r, V_r, m, n, pad)
            for s, sig in tqdm(zip(all_seeds, all_sigmas), total=len(all_seeds),
                               desc=f"  r={r:4d} all    ", leave=False)
        ]
        rho_spearman, p_val = spearmanr(all_rho, all_scores)

        results_by_rank[r] = {
            "mean_rho_plus":  mean_plus,
            "mean_rho_minus": mean_minus,
            "ratio":          ratio,
            "spearman_r":     float(rho_spearman),
            "spearman_p":     float(p_val),
            "rho_plus":       rho_plus,
            "rho_minus":      rho_minus,
        }

    # Decision gate
    best_ratio = max(v["ratio"] for v in results_by_rank.values())
    decision = "PROCEED_TO_PHASE2" if best_ratio >= 2.0 else "INVESTIGATE_ALTERNATIVE_BASIS"
    print(f"\nDecision gate: best ratio={best_ratio:.2f}x → {decision}")

    output = {
        "model":          args.model,
        "N":              args.N,
        "K":              args.K,
        "n_loc":          args.n_loc,
        "grad_norm":      float(grad.norm()),
        "results_by_rank": results_by_rank,
        "decision":       decision,
    }
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",          default="qwen-1.5b")
    parser.add_argument("--N",              type=int,   default=500)
    parser.add_argument("--K",              type=int,   default=50)
    parser.add_argument("--n-train",        type=int,   default=200)
    parser.add_argument("--n-loc",          type=int,   default=50,
                        help="D_loc size for gradient computation")
    parser.add_argument("--ranks",          type=int,   nargs="+",
                        default=[10, 50, 100, 200, 500])
    parser.add_argument("--batch-size",     type=int,   default=4)
    parser.add_argument("--seed",           type=int,   default=42)
    parser.add_argument("--phase0-results", default="results/phase0.json",
                        help="Reuse Phase 0 seed scores to skip re-running RandOpt")
    parser.add_argument("--output",         default="results/phase1.json")
    main(parser.parse_args())
