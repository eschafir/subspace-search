"""Countdown benchmark: reach a target number using arithmetic on given numbers."""
import re
import random
from typing import Optional
from datasets import load_dataset

SYSTEM_PROMPT = (
    "You are given a list of numbers and a target. "
    "Using each number at most once and the operations +, -, *, /, write an arithmetic "
    "expression that equals the target. Show your work, then write the final expression "
    "on its own line in the format: #### <expression>"
)

# Hugging Face dataset used by Neural Thickets for Countdown
_HF_DATASET = "Jiayi-Pan/Countdown-Tasks-3to4"


def load_countdown(n_train: int = 200, n_test: int = 200, seed: int = 42):
    """
    Load Countdown Tasks dataset and return (d_train, d_test).
    Each example: {'nums': [int, ...], 'target': int, 'question': str, 'answer': str}
    """
    ds = load_dataset(_HF_DATASET)
    rng = random.Random(seed)

    # This dataset may only have a 'train' split — adjust if needed
    train_pool = list(ds["train"])
    rng.shuffle(train_pool)

    # Use 80/20 split if no test split exists
    split = int(0.8 * len(train_pool))
    train_raw = train_pool[:split]
    test_raw  = train_pool[split:]

    def build(ex):
        nums = ex["nums"]
        target = ex["target"]
        question = f"Numbers: {nums}\nTarget: {target}"
        return {"question": question, "nums": nums, "target": target,
                "answer": ex.get("answer", "")}

    d_train = [build(ex) for ex in train_raw[:n_train]]
    d_test  = [build(ex) for ex in test_raw[:n_test]]
    return d_train, d_test


def format_prompt(question: str, tokenizer) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": question},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def extract_answer(text: str) -> Optional[str]:
    """Extract the expression after #### from model output."""
    match = re.search(r"####\s*(.+)", text)
    return match.group(1).strip() if match else None


def is_correct(generation: str, example: dict) -> bool:
    """Verify the extracted expression equals the target using only the given nums."""
    expr_str = extract_answer(generation)
    if expr_str is None:
        return False

    nums = list(example["nums"])
    target = example["target"]

    # Extract all numbers used in the expression
    used = [int(x) for x in re.findall(r"\d+", expr_str)]
    available = sorted(nums)
    if sorted(used) != available:
        return False

    try:
        result = eval(expr_str, {"__builtins__": {}})  # safe-ish for arithmetic
        return abs(result - target) < 1e-6
    except Exception:
        return False
