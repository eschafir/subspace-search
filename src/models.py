"""Model loading and in-place perturbation context manager."""
import contextlib
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def best_gpu() -> str:
    """
    Return the CUDA device string with the most free memory.
    Falls back to 'cpu' if CUDA is unavailable.

    Example: 'cuda:2'
    """
    if not torch.cuda.is_available():
        return "cpu"

    best_idx  = 0
    best_free = 0
    for i in range(torch.cuda.device_count()):
        free, _ = torch.cuda.mem_get_info(i)
        if free > best_free:
            best_free = free
            best_idx  = i

    device = f"cuda:{best_idx}"
    gb = best_free / 1024 ** 3
    print(f"  Selected {device}  ({gb:.1f} GB free"
          f" of {torch.cuda.device_count()} GPUs)")
    return device

MODELS = {
    "qwen-1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen-3b":   "Qwen/Qwen2.5-3B-Instruct",
    "qwen-32b":  "Qwen/Qwen2.5-32B-Instruct",
}


def load(name: str, device: str = "cuda", dtype=torch.bfloat16):
    model_id = MODELS.get(name, name)
    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # For large models like 32B, check if requested single GPU has enough free VRAM (~65GB)
    # If not, fall back to device_map="auto" across available GPUs
    dev_map = device
    if "32b" in name.lower() and isinstance(device, str) and device.startswith("cuda:"):
        gpu_idx = int(device.split(":")[1])
        free_bytes, _ = torch.cuda.mem_get_info(gpu_idx)
        free_gb = free_bytes / (1024 ** 3)
        if free_gb < 64.0:
            print(f"  Notice: {device} has {free_gb:.1f} GB free (< 64 GB needed for 32B model). Using device_map='auto'.")
            dev_map = "auto"
        
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=dtype, 
        device_map=dev_map,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa"  # Native PyTorch memory-efficient attention
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
