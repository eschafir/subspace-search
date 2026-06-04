"""GSM8K benchmark: loading, prompting, answer extraction, scoring."""
import re
import random
from typing import Optional
from datasets import load_dataset

SYSTEM_PROMPT = (
    "Solve the math problem step by step. "
    "At the end of your solution, write the final numeric answer on its own line "
    "in the format: #### <number>"
)


def load_gsm8k(n_train: int = 200, n_test: int = 200, seed: int = 42):
    """
    Load GSM8K and return (d_train, d_test) as lists of dicts with keys
    'question' and 'answer' (the reference answer string including #### line).
    """
    ds = load_dataset("openai/gsm8k", "main")
    rng = random.Random(seed)

    train_pool = list(ds["train"])
    test_pool  = list(ds["test"])

    rng.shuffle(train_pool)
    rng.shuffle(test_pool)

    d_train = [{"question": ex["question"], "answer": ex["answer"]}
               for ex in train_pool[:n_train]]
    d_test  = [{"question": ex["question"], "answer": ex["answer"]}
               for ex in test_pool[:n_test]]

    return d_train, d_test


def format_prompt(example: dict, tokenizer) -> str:
    """Build the chat-formatted prompt string (no answer, ready for generation)."""
    messages = [
        {"role": "system",  "content": SYSTEM_PROMPT},
        {"role": "user",    "content": example["question"]},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def format_full(example: dict, tokenizer) -> str:
    """Build full conversation (prompt + reference answer) for SFT gradient."""
    messages = [
        {"role": "system",    "content": SYSTEM_PROMPT},
        {"role": "user",      "content": example["question"]},
        {"role": "assistant", "content": example["answer"]},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )


def extract_answer(text: str) -> Optional[float]:
    """Extract the numeric answer from a model generation or reference string."""
    match = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", text)
    if match:
        return float(match.group(1).replace(",", ""))
    # Fallback: last number in the text
    nums = re.findall(r"-?[\d,]+(?:\.\d+)?", text)
    if nums:
        try:
            return float(nums[-1].replace(",", ""))
        except ValueError:
            pass
    return None


def get_reference_answer(example: dict) -> Optional[float]:
    return extract_answer(example["answer"])


def is_correct(predicted: Optional[float], reference: Optional[float]) -> bool:
    if predicted is None or reference is None:
        return False
    return abs(predicted - reference) < 0.5  # integer match with tolerance
