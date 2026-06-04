"""
Gradient subspace construction for Subspace RandOpt (§3.1.2, §3.2.2).

Global approach: flatten all trainable gradients into one vector g ∈ R^d,
reshape to G ∈ R^{m×n} (m ≈ n ≈ √d), compute truncated SVD to get U_r, V_r.

A perturbation in the subspace is Δ = σ · vec(U_r diag(z) V_r^T), z ~ N(0, I_r).
"""
import math
import torch
import torch.nn.functional as F
from typing import Tuple


def compute_gradient(
    model,
    tokenizer,
    examples: list[dict],
    device: str,
    format_full_fn,    # (example, tokenizer) -> str  (full conversation text)
    format_prompt_fn,  # (example, tokenizer) -> str  (prompt only)
) -> torch.Tensor:
    """
    Compute the mean SFT gradient on examples via one backward pass.
    Loss is restricted to answer tokens (prompt tokens masked with -100).

    Returns flat gradient vector on CPU in float32.
    Memory note: activations for backprop are ~3-4× model size.
    Enable gradient_checkpointing before calling on large models.
    """
    model.train()
    model.zero_grad()

    for ex in examples:
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
            (loss / len(examples)).backward()

    model.eval()

    grads = []
    for p in model.parameters():
        if p.requires_grad:
            g = (p.grad.detach().float() if p.grad is not None
                 else torch.zeros(p.numel(), dtype=torch.float32, device=device))
            grads.append(g.flatten())

    flat = torch.cat(grads).cpu()
    model.zero_grad()
    return flat  # shape (d,)


def build_subspace(
    grad: torch.Tensor,
    rank: int,
    device: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor, int, int, int]:
    """
    Build a rank-r gradient subspace from a flat gradient vector.

    Steps:
      1. Reshape grad ∈ R^d to G ∈ R^{m×n}  (m ≈ n ≈ √d, zero-padded)
      2. Truncated SVD via torch.svd_lowrank: U_r ∈ R^{m×r}, V_r ∈ R^{n×r}
      3. Return (U_r, V_r, m, n, pad) on CPU

    A subspace perturbation for a seed s is built by perturbation.make_subspace_delta.

    Memory: storing G requires ~d × 4 bytes (float32) → ~6 GB for 1.5B params.
    Use dtype=bfloat16 for G if memory is tight (quality nearly identical).
    """
    d = grad.numel()
    m = math.ceil(math.sqrt(d))
    n = math.ceil(d / m)
    pad = m * n - d

    g_padded = F.pad(grad.float(), (0, pad)) if pad > 0 else grad.float()
    G = g_padded.reshape(m, n).to(device)

    # Randomized truncated SVD: O(d · rank) time, O(d) memory
    # niter=4 gives good accuracy for well-conditioned gradients
    U, S, V = torch.svd_lowrank(G, q=rank, niter=4)

    return U.cpu(), V.cpu(), m, n, pad


def subspace_energy_ratio(
    delta: torch.Tensor,
    U_r: torch.Tensor,
    V_r: torch.Tensor,
    m: int,
    n: int,
    pad: int,
) -> float:
    """
    ρ_r(Δ) = ||P_r^T Δ||² / ||Δ||²

    For the Khatri-Rao subspace basis P_r (columns = vec(u_j ⊗ v_j)):
      (P_r^T Δ)_j = u_j^T · reshape(Δ, m, n) · v_j

    Computed as diag(U_r^T G_Δ V_r) without forming P_r explicitly.
    """
    d = delta.numel()
    assert d + pad == m * n, "delta length does not match subspace dimensions"

    g_padded = F.pad(delta.float(), (0, pad)) if pad > 0 else delta.float()
    G_delta = g_padded.reshape(m, n)  # (m, n)

    # coords[j] = u_j^T G_delta v_j
    coords = torch.einsum("mr,mn,nr->r", U_r, G_delta, V_r)  # (r,)

    proj_norm_sq = coords.pow(2).sum().item()
    total_norm_sq = delta.float().pow(2).sum().item()

    return proj_norm_sq / (total_norm_sq + 1e-12)
