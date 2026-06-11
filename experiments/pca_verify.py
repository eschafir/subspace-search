"""
PCA-based Subspace Verification.

This script tests whether the top-K perturbations found by RandOpt share a
low-dimensional solution subspace (thicket subspace).

Method:
  1. Load top-K and non-top seeds from Phase 0 checkpoint.
  2. Split top-K deltas into train (K_train) and validation (K_val) sets.
  3. Compute all pairwise dot products (Gram matrix) of top-K deltas on GPU.
  4. Perform PCA on the train set Gram matrix to find principal components.
  5. Project validation deltas and random non-top deltas onto the PCA subspace.
  6. Measure the mean energy ratio (ρ̄⁺ and ρ̄⁻) and alignment ratio.
"""
import argparse
import json
import os
import sys
import torch
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.models import load, param_count, best_gpu
from src.perturbation import make_isotropic_delta, assign_sigma
from src.metrics import alignment_ratio


def main(args):
    device = best_gpu()
    
    # Load Phase 0 results
    if not os.path.exists(args.phase0_results):
        print(f"Error: Phase 0 results not found at {args.phase0_results}")
        return
        
    print(f"Loading Phase 0 results from {args.phase0_results}...")
    with open(args.phase0_results) as f:
        p0 = json.load(f)
        
    top_seeds    = p0["top_seeds"][:args.K]
    top_sigmas   = p0["top_sigmas"][:args.K]
    all_seeds    = p0["all_seeds"]
    all_sigmas   = p0["all_sigmas"]
    
    # Determine param count d
    print(f"Loading model {args.model} to determine parameter count...")
    model, tokenizer = load(args.model, device=device)
    d = param_count(model)
    print(f"  d = {d:,} parameters")
    
    # Free model VRAM since we only need to construct perturbations using seeds
    del model
    torch.cuda.empty_cache()
    
    K = len(top_seeds)
    K_train = K // 2
    K_val = K - K_train
    
    train_idx = list(range(0, K, 2))[:K_train]
    val_idx = list(range(1, K, 2))[:K_val]
    
    print(f"\nComputing Gram matrix for top-K deltas (K={K}, train={K_train}, val={K_val})...")
    # Gram matrix K_mat of size K x K
    K_mat = torch.zeros(K, K, dtype=torch.float32)
    
    # Compute Gram matrix using GPU-accelerated dot products (memory-efficient)
    for i in tqdm(range(K), desc="Gram matrix rows"):
        delta_i = make_isotropic_delta(top_seeds[i], d, top_sigmas[i]).to(device)
        for j in range(i, K):
            delta_j = make_isotropic_delta(top_seeds[j], d, top_sigmas[j]).to(device)
            dot_val = torch.dot(delta_i, delta_j).item()
            K_mat[i, j] = dot_val
            K_mat[j, i] = dot_val
            del delta_j
        del delta_i
        torch.cuda.empty_cache()
        
    # Split Gram matrix
    K_train_mat = K_mat[train_idx][:, train_idx]
    
    # Perform SVD/Eigendecomposition on the training Gram matrix
    # K_train_mat = V S^2 V^T
    eigenvalues, V_train = torch.linalg.eigh(K_train_mat)
    
    # Sort descending
    idx = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[idx]
    V_train = V_train[:, idx]
    
    # Singular values of X_train
    s = torch.sqrt(torch.clamp(eigenvalues, min=1e-12))
    
    # Sample K_val non-top seeds for control group
    top_set = set(top_seeds)
    non_top_idx = [i for i, s in enumerate(all_seeds) if s not in top_set]
    rng = np.random.default_rng(args.seed)
    chosen = rng.choice(non_top_idx, size=K_val, replace=False)
    non_top_seeds  = [all_seeds[i]  for i in chosen]
    non_top_sigmas = [all_sigmas[i] for i in chosen]
    
    print("\nComputing cross dot products with non-top control deltas...")
    # X_train^T @ X_non_top (shape: K_train x K_val)
    K_non_top = torch.zeros(K_train, K_val, dtype=torch.float32)
    norm_non_top = torch.zeros(K_val, dtype=torch.float32)
    
    for j in tqdm(range(K_val), desc="Non-top deltas"):
        delta_nt = make_isotropic_delta(non_top_seeds[j], d, non_top_sigmas[j]).to(device)
        norm_non_top[j] = torch.dot(delta_nt, delta_nt).item()
        
        for i in range(K_train):
            delta_tr = make_isotropic_delta(top_seeds[train_idx[i]], d, top_sigmas[train_idx[i]]).to(device)
            K_non_top[i, j] = torch.dot(delta_tr, delta_nt).item()
            del delta_tr
            
        del delta_nt
        torch.cuda.empty_cache()
        
    # Validation projection coordinates in PCA basis: C = diag(1/s) V_train^T K_cross
    K_cross = K_mat[train_idx][:, val_idx]
    C_val = (1.0 / s.unsqueeze(1)) * (V_train.T @ K_cross)
    
    # Non-top projection coordinates
    C_non_top = (1.0 / s.unsqueeze(1)) * (V_train.T @ K_non_top)
    
    # Compute energy ratios for different ranks r
    ranks = [1, 2, 5, 10, 15, 20, 24]
    ranks = [r for r in ranks if r <= K_train]
    
    print("\nPCA Subspace Results:")
    print("----------------------------------------------------------------------")
    print(f"{'Rank r':<8} | {'ρ̄⁺_val (top-K)':<18} | {'ρ̄⁻_val (non-top)':<18} | {'Ratio':<8} | Status")
    print("----------------------------------------------------------------------")
    
    results_by_rank = {}
    for r in ranks:
        # Sum coords squared up to r
        proj_energy_val = torch.sum(C_val[:r, :].pow(2), dim=0)
        total_energy_val = torch.diagonal(K_mat[val_idx][:, val_idx])
        rho_plus_val = torch.mean(proj_energy_val / total_energy_val).item()
        
        proj_energy_nt = torch.sum(C_non_top[:r, :].pow(2), dim=0)
        rho_minus_val = torch.mean(proj_energy_nt / norm_non_top).item()
        
        ratio = alignment_ratio(rho_plus_val, rho_minus_val)
        status = "✓ STRONG" if ratio >= 2.0 else "✗ weak"
        
        print(f"{r:<8d} | {rho_plus_val:<18.3e} | {rho_minus_val:<18.3e} | {ratio:<8.2f}x | {status}")
        
        results_by_rank[r] = {
            "rho_plus_val": rho_plus_val,
            "rho_minus_val": rho_minus_val,
            "ratio": ratio
        }
        
    # Save results
    output_path = args.output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({
            "model": args.model,
            "K": args.K,
            "results_by_rank": results_by_rank
        }, f, indent=2)
    print(f"\nSaved results to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",          default="qwen-1.5b")
    parser.add_argument("--K",              type=int,   default=50)
    parser.add_argument("--seed",           type=int,   default=42)
    parser.add_argument("--phase0-results", default="results/phase0_checkpoint.json")
    parser.add_argument("--output",         default="results/pca_verify.json")
    main(parser.parse_args())
