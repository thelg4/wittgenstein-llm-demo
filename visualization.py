from __future__ import annotations

# Keep Transformers quiet
from transformers.utils import logging as hf_logging
hf_logging.set_verbosity_error()

from typing import List
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import torch
from sklearn.decomposition import PCA
from transformers import AutoTokenizer, AutoModel

from utils import get_device, cosine, contextual_embed_target

# ---------------- Config ----------------
MODEL_NAME = "bert-base-uncased"
TARGET_TOKEN = "bank"
SENTENCES = [
    "I sat by the river bank watching the sunset.",
    "I deposited my paycheck at the bank this morning.",
    "The memory bank stores all the computer's data.",
]
LABELS = ["river bank", "financial bank", "memory bank"]

FIG_W, FIG_H = 7.0, 5.6  # inches
DPI = 220
# ---------------------------------------


def set_matplotlib_defaults():
    """Readable, professional defaults."""
    mpl.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 16,
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "figure.dpi": DPI,
        "savefig.dpi": DPI,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlepad": 10,
    })


def plot_heatmap(sim: np.ndarray, labels: List[str]):
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), constrained_layout=True)
    im = ax.imshow(sim, vmin=0, vmax=1, aspect="equal")

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)

    # grid lines between cells
    ax.set_xticks(np.arange(-.5, len(labels), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(labels), 1), minor=True)
    ax.grid(which="minor", color="w", linewidth=1, alpha=0.9)
    ax.tick_params(which="minor", bottom=False, left=False)

    # value annotations
    for i in range(sim.shape[0]):
        for j in range(sim.shape[1]):
            ax.text(
                j, i, f"{sim[i, j]:.2f}",
                va="center", ha="center",
                color="white" if sim[i, j] > 0.5 else "black"
            )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("cosine similarity")

    ax.set_title("Contextual Similarity of 'bank'")
    fig.savefig("bank_cosine_heatmap.png", bbox_inches="tight")
    fig.savefig("bank_cosine_heatmap.svg", bbox_inches="tight")
    plt.close(fig)


def plot_pca(E: np.ndarray, labels: List[str]):
    XY = PCA(n_components=2, random_state=0).fit_transform(E)

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), constrained_layout=True)
    ax.scatter(XY[:, 0], XY[:, 1], s=70)

    for (x, y), lab in zip(XY, labels):
        ax.annotate(lab, (x, y), xytext=(6, 6), textcoords="offset points")

    ax.grid(True, alpha=0.25, linewidth=0.8)
    ax.set_aspect("equal", adjustable="datalim")

    pad = 0.1 * max(XY[:, 0].ptp(), XY[:, 1].ptp(), 1.0)
    ax.set_xlim(XY[:, 0].min() - pad, XY[:, 0].max() + pad)
    ax.set_ylim(XY[:, 1].min() - pad, XY[:, 1].max() + pad)

    ax.set_title("2D PCA of Contextual Embeddings for 'bank'")
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")

    fig.savefig("bank_pca_scatter.png", bbox_inches="tight")
    fig.savefig("bank_pca_scatter.svg", bbox_inches="tight")
    plt.close(fig)


def main():
    set_matplotlib_defaults()
    device = get_device()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device).eval()

    # contextual embeddings for the target token across sentences
    embs = [
        contextual_embed_target(tokenizer, model, s, TARGET_TOKEN, device)
        for s in SENTENCES
    ]
    E = torch.stack(embs).numpy()

    # cosine similarity matrix
    E_norm = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)
    sim = E_norm @ E_norm.T

    plot_heatmap(sim, LABELS)
    plot_pca(E, LABELS)

    print("Saved bank_cosine_heatmap.(png|svg) and bank_pca_scatter.(png|svg)")


if __name__ == "__main__":
    main()

