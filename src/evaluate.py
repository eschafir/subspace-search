"""
Score a (perturbed) model on a list of examples.

Two modes:
  - generation (default): autoregressive decode, extract answer, check correctness.
    Faithful to paper; ~0.5–2 hr for N=500 on D_train=200 with a 1.5B model.
  - loss: teacher-forced cross-entropy on answer tokens.
    ~10× faster; use as a proxy score for perturbation selection when generation is too slow.
"""
import torch
import torch.nn.functional as F
from tqdm import tqdm
from typing import Callable, List, Optional


# ---------------------------------------------------------------------------
# Generation-based scoring
# ---------------------------------------------------------------------------

def score_examples_generation(
    model,
    tokenizer,
    examples: list[dict],
    device: str,
    format_fn: Callable,
    correct_fn: Callable,
    max_new_tokens: int = 512,
    batch_size: int = 8,
) -> float:
    """
    Generate outputs for each example and return fraction correct.

    format_fn(example, tokenizer) -> prompt string
    correct_fn(generation_text, example) -> bool
    """
    n_correct = 0
    total = len(examples)

    for batch_start in range(0, total, batch_size):
        batch = examples[batch_start: batch_start + batch_size]
        prompts = [format_fn(ex, tokenizer) for ex in batch]

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        ).to(device)

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        # Decode only the newly generated tokens
        prompt_lengths = inputs["input_ids"].shape[1]
        for i, ex in enumerate(batch):
            new_tokens = out[i, prompt_lengths:]
            text = tokenizer.decode(new_tokens, skip_special_tokens=True)
            if correct_fn(text, ex):
                n_correct += 1

    return n_correct / total


# ---------------------------------------------------------------------------
# Loss-based scoring (faster proxy)
# ---------------------------------------------------------------------------

def score_examples_loss(
    model,
    tokenizer,
    examples: list[dict],
    device: str,
    format_full_fn: Callable,
    format_prompt_fn: Callable,
    batch_size: int = 8,
) -> float:
    """
    Compute mean negative cross-entropy loss on answer tokens (lower = better).
    Returns a score in (-inf, 0]; negate for use as a maximization objective.
    """
    total_loss = 0.0

    for batch_start in range(0, len(examples), batch_size):
        batch = examples[batch_start: batch_start + batch_size]

        full_texts   = [format_full_fn(ex, tokenizer)   for ex in batch]
        prompt_texts = [format_prompt_fn(ex, tokenizer) for ex in batch]

        full_enc   = tokenizer(full_texts,   return_tensors="pt", padding=True,
                               truncation=True, max_length=1024).to(device)
        prompt_enc = tokenizer(prompt_texts, return_tensors="pt", padding=True,
                               truncation=True, max_length=1024)

        with torch.no_grad():
            outputs = model(**full_enc, use_cache=False)
            logits  = outputs.logits  # (B, seq, vocab)

        for i in range(len(batch)):
            prompt_len = prompt_enc["input_ids"].shape[1]
            labels = full_enc["input_ids"][i].clone()
            labels[:prompt_len] = -100  # mask prompt

            # Shift for next-token prediction
            shift_logits = logits[i, :-1]
            shift_labels = labels[1:]

            loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)
            total_loss += loss.item()

    return -(total_loss / len(examples))  # negate → higher is better


# ---------------------------------------------------------------------------
# Majority vote over an ensemble of K models
# ---------------------------------------------------------------------------

def majority_vote_accuracy(
    model,
    tokenizer,
    examples: list[dict],
    device: str,
    seeds: list[int],
    sigmas: list[float],
    delta_fn: Callable,   # (seed, sigma) -> flat delta tensor
    format_fn: Callable,
    extract_fn: Callable,
    correct_fn: Callable,
    with_delta_ctx,
    max_new_tokens: int = 512,
) -> float:
    """
    For each test example, collect generations from K perturbed models and
    return the majority-voted answer accuracy.
    """
    from collections import Counter

    n_correct = 0
    for ex in tqdm(examples, desc="Majority vote"):
        prompt = format_fn(ex, tokenizer)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                           max_length=1024).to(device)

        answers = []
        for seed, sigma in zip(seeds, sigmas):
            delta = delta_fn(seed, sigma)
            with with_delta_ctx(model, delta):
                with torch.no_grad():
                    out = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id,
                    )
            text = tokenizer.decode(out[0, inputs["input_ids"].shape[1]:],
                                    skip_special_tokens=True)
            ans = extract_fn(text)
            answers.append(ans)

        voted = Counter(a for a in answers if a is not None)
        best  = voted.most_common(1)[0][0] if voted else None
        if correct_fn(best, ex):
            n_correct += 1

    return n_correct / len(examples)
