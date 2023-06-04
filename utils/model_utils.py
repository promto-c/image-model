import torch
import torch.nn as nn

def save_model(model: nn.Module, save_path: str) -> None:
    """Save a PyTorch model to a file using TorchScript and JIT.

    Args:
        model (nn.Module): The PyTorch model to save.
        save_path (str): The path to save the model.

    Returns:
        None
    """
    scripted_model = torch.jit.script(model)
    torch.jit.save(scripted_model, save_path)

def load_model(load_path: str) -> nn.Module:
    """Load a PyTorch model from a file using TorchScript and JIT.

    Args:
        load_path (str): The path to load the model from.

    Returns:
        nn.Module: The loaded PyTorch model.
    """
    scripted_model = torch.jit.load(load_path)
    return scripted_model
