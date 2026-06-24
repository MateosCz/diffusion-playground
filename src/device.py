import torch


def get_default_device() -> torch.device:
    """Return the best available torch device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_lightning_accelerator(device: torch.device | str) -> str:
    """Map a torch device to Lightning accelerator name."""
    resolved = torch.device(device)
    if resolved.type == "cuda":
        return "gpu"
    if resolved.type == "mps":
        return "mps"
    return "cpu"


def module_device(module: torch.nn.Module) -> torch.device:
    """Best-effort device lookup from module parameters or buffers."""
    for param in module.parameters():
        return param.device
    for buffer in module.buffers():
        return buffer.device
    return get_default_device()
