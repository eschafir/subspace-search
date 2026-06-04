"""Seed-based perturbation generation (isotropic and subspace-projected)."""
import torch
from typing import Tuple


def make_isotropic_delta(seed: int, d: int, sigma: float) -> torch.Tensor:
    """
    Generate a flat Gaussian perturbation ε ~ N(0, σ²I) from a seed.
    Returns CPU float32 tensor of length d.
    """
    g = torch.Generator()
    g.manual_seed(seed)
    return torch.randn(d, generator=g, dtype=torch.float32) * sigma


def make_subspace_delta(
    seed: int,
    U_r: torch.Tensor,  # (m, r)
    V_r: torch.Tensor,  # (n, r)
    m: int,
    n: int,
    pad: int,
    sigma: float,
) -> torch.Tensor:
    """
    Generate a subspace-projected perturbation from a seed.

    Samples z ~ N(0, I_r), constructs ΔG = U_r diag(z) V_r^T ∈ R^{m×n},
    then returns vec(ΔG)[:-pad] * sigma as a flat CPU float32 tensor.

    This confines all perturbation energy to the gradient subspace S_r
    while producing a full-dimensional parameter-space vector.
    """
    r = U_r.shape[1]
    g = torch.Generator()
    g.manual_seed(seed)
    z = torch.randn(r, generator=g, dtype=torch.float32)  # (r,)

    # (m, r) * (r,) elementwise then @ (r, n) — equivalent to U_r diag(z) V_r^T
    delta_mat = (U_r * z.unsqueeze(0)) @ V_r.T  # (m, n)
    delta_flat = delta_mat.flatten()

    if pad > 0:
        delta_flat = delta_flat[:-pad]

    return delta_flat * sigma


def assign_sigma(seed_idx: int, sigmas: list[float], N: int) -> float:
    """Round-robin assignment of sigma values across N seeds."""
    n_per_sigma = max(1, N // len(sigmas))
    return sigmas[min(seed_idx // n_per_sigma, len(sigmas) - 1)]
