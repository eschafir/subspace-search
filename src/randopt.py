"""
Vanilla RandOpt — Algorithm 1 from Neural Thickets (Gan & Isola, 2026).

Training phase:  sample N Gaussian perturbations, score each on D_train,
                 keep top-K seeds.
Inference phase: majority vote over K perturbed models.
"""
import torch
from tqdm import tqdm
from typing import Callable

from src.models import with_delta, param_count
from src.perturbation import make_isotropic_delta, assign_sigma


def score_perturbation(
    model,
    tokenizer,
    examples: list[dict],
    device: str,
    delta: torch.Tensor,
    score_fn: Callable,      # score_fn(model, tokenizer, examples, device) -> float
) -> float:
    """Temporarily apply delta and return the score on examples."""
    with with_delta(model, delta):
        return score_fn(model, tokenizer, examples, device)


def randopt(
    model,
    tokenizer,
    d_train: list[dict],
    N: int,
    K: int,
    sigmas: list[float],
    device: str,
    score_fn: Callable,
    base_seed: int = 0,
) -> dict:
    """
    Run vanilla RandOpt selection phase.

    Returns a dict with:
      top_seeds   : list[int]   (length K)
      top_sigmas  : list[float] (length K)
      top_scores  : list[float] (length K, descending)
      all_scores  : list[float] (length N)
      all_seeds   : list[int]   (length N)
      all_sigmas  : list[float] (length N)
    """
    d = param_count(model)
    all_scores  = []
    all_seeds   = []
    all_sigmas  = []

    for i in tqdm(range(N), desc="RandOpt sampling"):
        seed  = base_seed + i
        sigma = assign_sigma(i, sigmas, N)
        delta = make_isotropic_delta(seed, d, sigma)

        score = score_perturbation(model, tokenizer, d_train, device, delta, score_fn)

        all_scores.append(score)
        all_seeds.append(seed)
        all_sigmas.append(sigma)

    # Select top-K by score (descending)
    ranked = sorted(zip(all_scores, all_seeds, all_sigmas), reverse=True)
    top_k  = ranked[:K]

    return {
        "top_seeds":  [s for _, s, _ in top_k],
        "top_sigmas": [σ for _, _, σ in top_k],
        "top_scores": [v for v, _, _ in top_k],
        "all_scores": all_scores,
        "all_seeds":  all_seeds,
        "all_sigmas": all_sigmas,
    }


def subspace_randopt(
    model,
    tokenizer,
    d_train: list[dict],
    N: int,
    K: int,
    sigma: float,
    device: str,
    score_fn: Callable,
    U_r: torch.Tensor,
    V_r: torch.Tensor,
    m: int,
    n: int,
    pad: int,
    base_seed: int = 0,
) -> dict:
    """
    Subspace RandOpt selection phase — Algorithm 2.

    Perturbations are confined to the gradient subspace S_r instead of R^d.
    Interface identical to randopt(); use the same top-K selection and
    majority_vote_gsm8k for inference.
    """
    from src.perturbation import make_subspace_delta

    all_scores = []
    all_seeds  = []

    for i in tqdm(range(N), desc="Subspace RandOpt sampling"):
        seed  = base_seed + i
        delta = make_subspace_delta(seed, U_r, V_r, m, n, pad, sigma)
        score = score_perturbation(model, tokenizer, d_train, device, delta, score_fn)
        all_scores.append(score)
        all_seeds.append(seed)

    ranked = sorted(zip(all_scores, all_seeds), reverse=True)
    top_k  = ranked[:K]

    return {
        "top_seeds":  [s for _, s in top_k],
        "top_sigmas": [sigma] * K,
        "top_scores": [v for v, _ in top_k],
        "all_scores": all_scores,
        "all_seeds":  all_seeds,
    }
