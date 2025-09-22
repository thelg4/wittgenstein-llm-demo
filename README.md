# Wittgenstein x LLMs: Contextual Embeddings Demo

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

```
git clone https://github.com/YOUR-USER/wittgenstein-llm-demo.git
cd wittgenstein-llm-demo

python3 -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

Run the demo:

```
python demo.py
```

Example output:
```
River bank vs Financial bank: 0.412
River bank vs Memory bank:    0.389
Financial bank vs Memory bank:0.534
```

---

## Visuals (optional)

Generate a heatmap and 2D projection:

```
python viz.py
```

This will create:
- `bank_cosine_heatmap.png`
- `bank_pca_scatter.png`

You can embed these images in blog posts, slides, or just open them locally.

---

## Requirements

- Python 3.9+  
- Dependencies are listed in `requirements.txt`:
  - `torch`
  - `transformers`
  - `numpy`
  - `scikit-learn`
  - `matplotlib`

