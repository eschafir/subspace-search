"""MBPP benchmark: code generation with unit test execution."""
import re
import random
import contextlib
import io
import textwrap
from typing import Optional
from datasets import load_dataset

SYSTEM_PROMPT = (
    "Write a Python function to solve the problem. "
    "Output only the function definition, no explanations."
)


def load_mbpp(n_train: int = 200, n_test: int = 200, seed: int = 42):
    """
    Load MBPP and return (d_train, d_test).
    Each example: {'question': str, 'tests': [str], 'code': str}
    """
    ds = load_dataset("google-research-datasets/mbpp", "sanitized")
    rng = random.Random(seed)

    train_pool = list(ds["train"])
    test_pool  = list(ds["test"])
    rng.shuffle(train_pool)
    rng.shuffle(test_pool)

    def build(ex):
        return {
            "question": ex["text"],
            "tests":    ex["test_list"],
            "code":     ex["code"],
        }

    d_train = [build(ex) for ex in train_pool[:n_train]]
    d_test  = [build(ex) for ex in test_pool[:n_test]]
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
    """Extract the first Python code block from model output."""
    match = re.search(r"```python\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Fallback: assume entire output is code
    return text.strip()


def is_correct(generation: str, example: dict, timeout: float = 5.0) -> bool:
    """Execute generated code + unit tests in a restricted environment."""
    code = extract_answer(generation)
    if code is None:
        return False

    test_code = "\n".join(example["tests"])
    full_code = textwrap.dedent(code) + "\n" + textwrap.dedent(test_code)

    namespace: dict = {}
    try:
        exec(compile(full_code, "<mbpp>", "exec"), namespace)
        return True
    except Exception:
        return False
