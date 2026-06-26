# miniGPT-from-scratch

A complete Transformer language model built from scratch — no HuggingFace.

## Repository Structure

```
miniGPT-from-scratch/
├── tokenizer/
│   ├── bpe.py             # BPE training algorithm
│   └── encode_decode.py   # Tokenizer (encode / decode)
├── model/
│   ├── attention.py       # Self-attention, multi-head attention
│   ├── transformer.py     # Full GPT block + MiniGPT model
│   └── positional.py      # Sinusoidal positional encoding
├── training/
│   ├── train.py           # Training loop (AdamW / SGD + cosine LR)
│   ├── dataset.py         # Data loading + DataLoader construction
│   └── config.py          # All hyperparameters + ablation grid
├── experiments/
│   └── run_ablations.py   # Full experiment grid + auto-generated plots
├── data/                  # shakespeare.txt auto-downloaded on first run
├── reports/               # Saved graphs + ablation_results.json
├── miniGPT_from_scratch.ipynb
└── README.md
```

## Quick Start

```bash
# 1. Install dependencies
pip install torch numpy matplotlib requests

# 2. Train baseline model
python -m training.train

# 3. Run full ablation grid
python -m experiments.run_ablations

# 4. Open notebook
jupyter lab miniGPT_from_scratch.ipynb
```

## Architecture

- **BPE Tokenizer** — trained from scratch, vocab size 1500
- **Sinusoidal Positional Encoding** — fixed, no extra parameters
- **Multi-Head Self-Attention** — causal mask, weight-tied output projection
- **Pre-LayerNorm Transformer Blocks** — more stable than post-LN
- **FeedForward** — 2-layer MLP with GELU activation

## Research Question

> *How does model size affect reasoning ability in small language models?*

See `reports/` for training curves and the paper-style report in the notebook.

## References

- Vaswani et al. (2017) — Attention Is All You Need
- Radford et al. (2019) — GPT-2
- Sennrich et al. (2016) — BPE
- Hoffmann et al. (2022) — Chinchilla scaling laws
