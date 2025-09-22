from typing import List
import torch
import torch.nn.functional as F

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def find_token_indices(tokens: List[str], target: str) -> List[int]:
    """
    Return indices where token equals `target`.
    With BERT-uncased, 'bank' typically appears as 'bank'.
    This keeps it robust if it appears multiple times.
    """
    return [i for i, t in enumerate(tokens) if t == target]

def cosine(u: torch.Tensor, v: torch.Tensor) -> float:
    u = F.normalize(u.unsqueeze(0), dim=1)
    v = F.normalize(v.unsqueeze(0), dim=1)
    return float((u @ v.T).item())
