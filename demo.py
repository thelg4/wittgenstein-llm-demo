from __future__ import annotations

import re
import logging
import warnings
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from transformers.utils import logging as hf_logging


# ================== QUIET SETUP ==================
def quiet_transformers() -> None:
    """Reduce Hugging Face / transformers noise without hiding real errors."""
    hf_logging.set_verbosity_error()
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
    # Hide only the noisy LayerNorm rename notices (gamma/beta → weight/bias)
    warnings.filterwarnings(
        "ignore",
        message=r"A parameter name that contains `(?:beta|gamma)` will be renamed internally",
        category=UserWarning,
    )


# ================== CONFIG ==================
MODEL_NAME = "bert-base-uncased"
TARGET = "bank"
SENTENCES = [
    "I sat by the river bank watching the sunset.",
    "I deposited my paycheck at the bank this morning.",
    "The memory bank stores all the computer's data.",
]


# ================== UTILITIES ==================
def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class Encoded:
    input_ids: torch.Tensor          # [1, seq_len]
    attention_mask: torch.Tensor     # [1, seq_len]
    offsets: torch.Tensor            # [1, seq_len, 2]


def encode(
    tokenizer: AutoTokenizer, text: str
) -> Encoded:
    enc = tokenizer(
        text,
        return_tensors="pt",
        return_offsets_mapping=True,
        add_special_tokens=True,
    )
    # Some tokenizers return offsets as LongTensor; keep as tensor for indexing
    return Encoded(
        input_ids=enc["input_ids"],
        attention_mask=enc["attention_mask"],
        offsets=enc["offset_mapping"],
    )


def find_word_spans(text: str, target: str) -> List[Tuple[int, int]]:
    """
    Find character spans for whole-word matches of `target` (case-insensitive).
    Uses \b word boundaries so 'bank' won't match 'banking'.
    """
    pattern = re.compile(rf"\b{re.escape(target)}\b", flags=re.IGNORECASE)
    return [m.span() for m in pattern.finditer(text)]


def token_indices_for_spans(offsets: torch.Tensor, spans: List[Tuple[int, int]]) -> List[int]:
    """
    Given offsets (seq_len, 2) and a list of (start, end) character spans,
    return the token indices whose offsets overlap any span.
    Skips special tokens which usually have (0, 0) or empty spans.
    """
    idxs: List[int] = []
    offs = offsets[0].tolist()  # [[start, end], ...]
    for i, (a, b) in enumerate(offs):
        if a == b:  # special tokens like [CLS]/[SEP]
            continue
        for (s, e) in spans:
            # overlap if not disjoint
            if not (b <= s or a >= e):
                idxs.append(i)
                break
    return idxs


def cosine(u: torch.Tensor, v: torch.Tensor) -> float:
    u = F.normalize(u.unsqueeze(0), dim=1)
    v = F.normalize(v.unsqueeze(0), dim=1)
    return float((u @ v.T).item())


# ================== CORE ==================
def contextual_embed_target(
    tokenizer: AutoTokenizer,
    model: AutoModel,
    device: torch.device,
    sentence: str,
    target: str,
) -> torch.Tensor:
    """
    Return a contextual embedding for all tokens corresponding to `target`
    (averaged) using the average of the last 4 hidden layers for stability.
    """
    enc = encode(tokenizer, sentence)
    spans = find_word_spans(sentence, target)
    if not spans:
        raise ValueError(f"Target '{target}' not found as a whole word in: {sentence}")

    with torch.no_grad():
        out = model(
            input_ids=enc.input_ids.to(device),
            attention_mask=enc.attention_mask.to(device),
            output_hidden_states=True,
        )

    # hidden_states: tuple(len = num_layers + 1), each [1, seq_len, hidden]
    last4 = torch.stack(out.hidden_states[-4:]).mean(0)  # [1, seq_len, hidden]
    idxs = token_indices_for_spans(enc.offsets, spans)

    if not idxs:
        # Fallback: if tokenizer split oddly, report tokens for debug
        toks = tokenizer.convert_ids_to_tokens(enc.input_ids[0].tolist())
        raise ValueError(
            f"No token indices overlapped '{target}' spans.\n"
            f"Sentence: {sentence}\nTokens: {toks}\nOffsets: {enc.offsets}"
        )

    emb = last4[0, idxs, :].mean(dim=0)  # [hidden]
    return emb.cpu()


def main() -> None:
    quiet_transformers()
    device = get_device()
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    model = AutoModel.from_pretrained(MODEL_NAME, from_tf=False).to(device).eval()

    # If you want more deterministic demos, uncomment:
    # torch.manual_seed(0)
    # if device.type == "cuda":
    #     torch.cuda.manual_seed_all(0)

    # Compute embeddings
    embs = [
        contextual_embed_target(tokenizer, model, device, s, TARGET)
        for s in SENTENCES
    ]

    # Pairwise cosine similarities
    sim_river_financial = cosine(embs[0], embs[1])
    sim_river_memory    = cosine(embs[0], embs[2])
    sim_financial_memory = cosine(embs[1], embs[2])

    print(f"River vs Financial: {sim_river_financial:.3f}")
    print(f"River vs Memory:    {sim_river_memory:.3f}")
    print(f"Financial vs Memory:{sim_financial_memory:.3f}")

    print(
        "\nNote: Exact values vary by run/model/hardware; the pattern to look for is "
        "'financial vs memory' being closer than either is to 'river'."
    )


if __name__ == "__main__":
    torch.set_grad_enabled(False)  # extra safety; we're not training
    main()
