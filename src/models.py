"""Model loading and in-place perturbation context manager."""
import contextlib
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODELS = {
    "qwen-1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen-3b":   "Qwen/Qwen2.5-3B-Instruct",
}


def load(name: str, device: str = "cuda", dtype=torch.bfloat16):
    model_id = MODELS.get(name, name)
    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype, device_map=device
    )
    model.eval()
    return model, tokenizer


def trainable_params(model):
    return [p for p in model.parameters() if p.requires_grad]


def param_count(model) -> int:
    return sum(p.numel() for p in trainable_params(model))


def flat_params(model) -> torch.Tensor:
    """Return a flat copy of all trainable parameters (CPU, float32)."""
    return torch.cat([p.detach().float().flatten().cpu() for p in trainable_params(model)])


@contextlib.contextmanager
def with_delta(model, delta: torch.Tensor):
    """
    Temporarily add a flat delta vector to all trainable parameters.

    delta is a CPU float32 tensor of length param_count(model).
    Parameters are modified in-place and restored on exit — thread-unsafe.
    """
    params = trainable_params(model)
    offset = 0
    for p in params:
        n = p.numel()
        chunk = delta[offset: offset + n].reshape(p.shape).to(dtype=p.dtype, device=p.device)
        p.data.add_(chunk)
        offset += n
    try:
        yield model
    finally:
        offset = 0
        for p in params:
            n = p.numel()
            chunk = delta[offset: offset + n].reshape(p.shape).to(dtype=p.dtype, device=p.device)
            p.data.sub_(chunk)
            offset += n
