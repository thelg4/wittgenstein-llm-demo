from __future__ import annotations
from transformers.utils import logging as hf_logging
hf_logging.set_verbosity_error()  # options: debug/info/warning/error

# You can also do this via environment (works even before Python runs):
#   export TRANSFORMERS_VERBOSITY=error

# ============== STANDARD IMPORTS ==============
import logging
from typing import List
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

# Optional: if you want to see *your* prints but not library noise
logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("demo")


# ============== CONFIG (EDIT ME) ==============
MODEL_NAME = "bert-base-uncased"
TARGET_TOKEN = "bank"  # what token we want the contextual embedding for
SENTENCES = [
    "I sat by the river bank watching the sunset.",
    "I deposited my paycheck at the bank this morning.",
    "The memory bank stores all the computer's data.",
]

# If you need deterministic-ish behavior for tutorials (not required here),
# uncomment the following lines:
# torch.manual_seed(0)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed_all(0)


# ============== SMALL UTILITIES ==============
def get_device() -> torch.device:
    """
    Pick the best available device:
      - CUDA GPU if available
      - Apple Silicon MPS if available
      - else CPU
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def find_token_indices(tokens: List[str], target: str) -> List[int]:
    """
    Return ALL indices where the token equals `target`.
    Notes:
      - With `bert-base-uncased`, "bank" typically appears exactly as "bank".
      - WordPiece sub-tokens are prefixed with "##" for suffix fragments.
      - For robustness, we strictly match the token "bank" (not "##bank").
    """
    return [i for i, t in enumerate(tokens) if t == target]


def cosine(u: torch.Tensor, v: torch.Tensor) -> float:
    """
    Cosine similarity between two 1-D vectors (PyTorch tensors).
    We L2-normalize first to be explicit/robust.
    """
    u = F.normalize(u.unsqueeze(0), dim=1)
    v = F.normalize(v.unsqueeze(0), dim=1)
    return float((u @ v.T).item())


# ============== CORE LOGIC ==============
def embed_target(
    tokenizer: AutoTokenizer,
    model: AutoModel,
    sentence: str,
    target: str,
    device: torch.device,
) -> torch.Tensor:
    """
    Compute the contextual embedding for `target` in `sentence`.

    How it works:
      1) Tokenize the sentence (adds [CLS]/[SEP] and subword pieces).
      2) Run the model to get the last hidden states: shape [1, seq_len, hidden_size].
      3) Convert input IDs back to tokens so we can find the positions of `target`.
      4) If `target` occurs multiple times (rare but possible), we average those positions.

    Why average multiple occurrences?
      - It's a simple, robust aggregation that avoids picking an arbitrary occurrence.
    """
    # Tokenize and move to device
    inputs = tokenizer(sentence, return_tensors="pt").to(device)

    # Forward pass (no grads needed for inference)
    with torch.no_grad():
        outputs = model(**inputs)  # BaseModelOutput(last_hidden_state=...)

    # Last hidden state for each token position (including special tokens)
    last_hidden = outputs.last_hidden_state  # [1, seq_len, hidden_size]

    # Map IDs -> readable tokens so we can locate "bank"
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0].tolist())

    # Find indices of the *exact* token "bank"
    idxs = find_token_indices(tokens, target)

    # If nothing found, this usually means punctuation or tokenization mismatch.
    # We fail loudly so readers learn what's happening.
    if not idxs:
        raise ValueError(
            f"Token '{target}' not found in tokenized sequence.\n"
            f"Sentence: {sentence}\n"
            f"Tokens:   {tokens}"
        )

    # Average across occurrences (shape: [hidden_size])
    emb = last_hidden[0, idxs, :].mean(dim=0)

    # Return on CPU for easy downstream math/printing regardless of device
    return emb.detach().cpu()


def main():
    # 1) Pick device
    device = get_device()
    log.info(f"Using device: {device}")

    # 2) Load tokenizer & model
    #    The logger is already set to ERROR, so noisy rename warnings won't show.
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()  # explicit: we're not training

    # 3) Embed target token in each sentence
    embeddings = []
    for s in SENTENCES:
        emb = embed_target(tokenizer, model, s, TARGET_TOKEN, device)
        embeddings.append(emb)

    # 4) Pairwise cosine similarities (the "story" this demo tells)
    sim_river_financial = cosine(embeddings[0], embeddings[1])
    sim_river_memory = cosine(embeddings[0], embeddings[2])
    sim_financial_memory = cosine(embeddings[1], embeddings[2])

    # 5) Print clean, nicely aligned results
    print(f"River bank vs Financial bank: {sim_river_financial:.3f}")
    print(f"River bank vs Memory bank:    {sim_river_memory:.3f}")
    print(f"Financial bank vs Memory bank:{sim_financial_memory:.3f}")

    # Tip for readers:
    # Values need not match exactly run-to-run, but the *pattern* should persist:
    # - financial vs memory often > river vs memory (both 'abstract/institutional' senses)
    # - river vs financial typically the lowest (concrete geography vs institution)


if __name__ == "__main__":
    main()