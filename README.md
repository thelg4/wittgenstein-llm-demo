# Wittgenstein x LLMs — Contextual Embeddings Demo

> “For a large class of cases—though not for all—the meaning of a word is its use in the language.” — *Wittgenstein, Philosophical Investigations §43*

This repo accompanies my article *Language Games and Language Models: Wittgenstein’s Philosophy in the Age of LLMs*.  
It demonstrates how modern transformer models represent meaning in use, not as fixed dictionary definitions.

---

## What this demo shows
- Loads **BERT (bert-base-uncased)** via Hugging Face.
- Embeds the word **“bank”** in three different contexts:
  1. River bank  
  2. Financial bank  
  3. Memory bank  
- Computes pairwise **cosine similarities** to show how usage changes meaning.
- Optionally generates **visuals**: a cosine-similarity heatmap and a 2D PCA projection.

---

## Quickstart

Clone the repo and set up a virtual environment:

```bash
git clone https://github.com/YOUR-USER/wittgenstein-llm-demo.git
cd wittgenstein-llm-demo

python3 -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

Run the demo:

```bash
python demo.py
```

Example output:
```
Using device: cpu
River vs Financial: 0.471
River vs Memory:    0.477
Financial vs Memory:0.537
```

Exact values vary by hardware (CPU/GPU/MPS) and library versions, but the **pattern** is consistent:  
*Financial vs Memory* is the closest pair, while *River vs Financial* and *River vs Memory* are farther apart.

> ⚠️ Note: On macOS you may see a `NotOpenSSLWarning` from `urllib3` if Python is linked against LibreSSL.  
> It does **not** affect results. To silence it, either pin `urllib3<2` or upgrade your Python/OpenSSL.

---

## Visuals (optional)

Generate a heatmap and 2D projection:

```bash
python viz.py
```

This will create:
- `bank_cosine_heatmap.png`
- `bank_cosine_heatmap.svg`
- `bank_pca_scatter.png`
- `bank_pca_scatter.svg`

These figures clearly show how contextual embeddings cluster by sense:
- the heatmap highlights relative cosine similarities,  
- the PCA scatter shows the three “banks” in 2D space.

---

## Repo contents

```
.
├── demo.py                # Minimal script: compute and compare embeddings
├── utils.py               # Shared helpers (device, cosine, token spans)
├── viz.py                 # Generate heatmap + PCA scatter
├── requirements.txt       # Dependencies
├── README.md              # This file
├── bank_cosine_heatmap.png/svg   # Example outputs
└── bank_pca_scatter.png/svg
```

---

## Requirements

- Python 3.9+  
- Dependencies are listed in `requirements.txt`:
  - `torch`
  - `transformers`
  - `numpy`
  - `scikit-learn`
  - `matplotlib`