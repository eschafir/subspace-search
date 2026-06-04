"""
Diagnostic metrics from both papers.

  subspace_energy_ratio  — ρ_r(Δ), measures projection onto gradient subspace
  spectral_discordance   — 𝒟, measures diversity of specialist perturbations
"""
import numpy as np
import torch
from src.subspace import subspace_energy_ratio as _rho


def mean_subspace_energy(
    deltas: list[torch.Tensor],
    U_r: torch.Tensor,
    V_r: torch.Tensor,
    m: int,
    n: int,
    pad: int,
) -> float:
    """Mean ρ_r across a list of delta vectors."""
    return float(np.mean([_rho(d, U_r, V_r, m, n, pad) for d in deltas]))


def spectral_discordance(score_matrix: np.ndarray) -> float:
    """
    𝒟 = 1 - (1 / M(M-1)) Σ_{j≠k} C_{jk}

    score_matrix: (N, M)  N seeds × M tasks, values are raw scores (any scale).
    Each column is converted to percentile ranks before computing Pearson correlation.

    Returns 𝒟 ∈ [0, M/(M-1)].
    """
    N, M = score_matrix.shape
    if M < 2:
        raise ValueError("spectral_discordance requires at least 2 tasks")

    # Convert to percentile ranks (column-wise)
    from scipy.stats import rankdata
    P = np.stack([rankdata(score_matrix[:, j]) / N for j in range(M)], axis=1)

    # Pearson correlation matrix of task columns
    C = np.corrcoef(P.T)  # (M, M)

    # Off-diagonal mean
    mask = ~np.eye(M, dtype=bool)
    off_diag_mean = C[mask].mean()

    return float(1.0 - off_diag_mean)


def alignment_ratio(rho_plus: float, rho_minus: float) -> float:
    """ρ̄⁺ / ρ̄⁻ — the decision-gate ratio from §3.1.4. >2× → proceed to Phase 2."""
    return rho_plus / (rho_minus + 1e-12)
