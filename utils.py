from __future__ import annotations
import re
from typing import List, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel


# ---------------- Device / math ----------------
def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def cosine(u: torch.Tensor, v: torch.Tensor) -> float:
    u = F.normalize(u.unsqueeze(0), dim=1)
    v = F.normalize(v.unsqueeze(0), dim=1)
    return float((u @ v.T).item())


# ---------------- Tokenization helpers ----------------
def find_word_spans(text: str, target: str) -> List[Tuple[int, int]]:
    """Whole-word, case-insensitive spans (won't match 'banking')."""
    pat = re.compile(rf"\b{re.escape(target)}\b", flags=re.IGNORECASE)
    return [m.span() for m in pat.finditer(text)]


def token_indices_for_spans(offsets: torch.Tensor, spans: List[Tuple[int, int]]) -> List[int]:
    """Map char spans → token indices via offset_mapping; skip specials."""
    idxs: List[int] = []
    offs = offsets[0].tolist()  # [[start, end], ...]
    for i, (a, b) in enumerate(offs):
        if a == b:  # special tokens like [CLS]/[SEP]
            continue
        for (s, e) in spans:
            if not (b <= s or a >= e):  # overlap
                idxs.append(i)
                break
    return idxs


# ---------------- Core embedding util ----------------
def contextual_embed_target(
    tokenizer: AutoTokenizer,
    model: AutoModel,
    sentence: str,
    target: str,
    device: torch.device,
) -> torch.Tensor:
    """
    Contextual embedding for `target` in `sentence`.
    Uses fast tokenizer + offset_mapping; averages last 4 layers; averages over multiple occurrences.
    Returns a 1-D tensor [hidden].
    """
    enc = tokenizer(sentence, return_tensors="pt", return_offsets_mapping=True)
    spans = find_word_spans(sentence, target)
    if not spans:
        raise ValueError(f"Target '{target}' not found as a whole word in: {sentence}")

    with torch.no_grad():
        out = model(
            **{k: v.to(device) if hasattr(v, "to") else v for k, v in enc.items() if k != "offset_mapping"},
            output_hidden_states=True,
        )

    # Average last 4 hidden layers for stability
    hidden = torch.stack(out.hidden_states[-4:]).mean(0)[0]  # [seq_len, hidden]
    idxs = token_indices_for_spans(enc["offset_mapping"], spans)
    if not idxs:
        toks = tokenizer.convert_ids_to_tokens(enc["input_ids"][0].tolist())
        raise ValueError(f"No token indices overlapped '{target}'. Tokens: {toks}")

    return hidden[idxs].mean(0).cpu()
