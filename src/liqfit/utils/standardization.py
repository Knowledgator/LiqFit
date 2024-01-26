from __future__ import annotations
from typing import List, Tuple
import torch
import numpy as np


def convert_to_numpy(x: torch.Tensor | Tuple | List | np.ndarray) -> np.ndarray:
    """Converts torch.Tensor, Tuple, List or NumPy array to Numpy Array.

    Args:
        x (torch.Tensor | Tuple | List | np.ndarray): Input to convert to
            NumPy array.

    Returns:
        np.ndarray: Converted NumPy array.
    """
    if isinstance(x, torch.tensor):
        return x.detach().cpu().numpy()
    else:
        return np.array(x)


def convert_to_torch(x: torch.Tensor | Tuple | List | np.ndarray) -> torch.Tensor:
    """Converts input to torch.Tensor

    Args:
        x (torch.Tensor | Tuple | List | np.ndarray): _description_

    Raises:
        ValueError: If the input is not a type of `torch.Tensor`,
            `Tuple`, `List`, `np.ndarray`

    Returns:
        torch.Tensor: Converted torch.Tensor.
    """
    if isinstance(x, (list, tuple)):
        return torch.tensor(x)
    elif isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    elif isinstance(x, torch.Tensor):
        return x
    else:
        raise ValueError(
            "Expected `List`, `Tuple` or `np.ndarray`. "
            f"Received: {type(x)}."
        )
