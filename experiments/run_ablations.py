"""
experiments/run_ablations.py
─────────────────────────────────────────────────────────────────────────────
Runs the full ablation grid defined in training/config.py and produces:
  • training_curves.png
  • model_size_vs_perplexity.png
  • attention_heatmap.png
  • lr_schedule.png
  • ablation_results.json

Run:
  python -m experiments.run_ablations
─────────────────────────────────────────────────────────────────────────────
"""
import torch

print("CUDA available:", torch.cuda.is_available())
print("CUDA devices:", torch.cuda.device_count())

if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
import json
import math
import os
import random
import sys

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

# allow running from repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from model.transformer import MiniGPT
from tokenizer.encode_decode import Tokenizer
from training.config import ABLATION_GRID, TokenizerConfig, TrainConfig
from training.dataset import fetch_shakespeare, build_tokenizer, build_dataloaders
from training.train import train


# ── setup ─────────────────────────────────────────────────────────────────────

SEED   = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTDIR = "reports"
os.makedirs(OUTDIR, exist_ok=True)

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]


# ── generation helper ─────────────────────────────────────────────────────────

@torch.no_grad()
def generate_sample(
    model:      MiniGPT,
    tokenizer:  Tokenizer,
    prompt:     str = "ROMEO:",
    max_new:    int = 50,
    temperature: float = 0.9,
) -> str:
    model.eval()
    ids = tokenizer.encode(prompt)
    idx = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(DEVICE)
    out = model.generate(idx, max_new_tokens=max_new, temperature=temperature)
    return tokenizer.decode(out[0].tolist())


# ── plot helpers ──────────────────────────────────────────────────────────────

def plot_training_curves(all_histories: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("miniGPT — Training Curves", fontsize=16, fontweight="bold")

    for i, (name, hist) in enumerate(all_histories.items()):
        c = COLORS[i % len(COLORS)]
        ep = range(1, len(hist["train_loss"]) + 1)
        axes[0].plot(ep, hist["train_loss"], color=c, label=name, lw=2)
        axes[1].plot(ep, hist["val_loss"],   color=c, label=name, lw=2)
        axes[2].plot(ep, hist["val_ppl"],    color=c, label=name, lw=2)

    titles   = ["Train Loss", "Validation Loss", "Validation Perplexity"]
    ylabels  = ["Cross-Entropy", "Cross-Entropy", "Perplexity"]
    for ax, t, yl in zip(axes, titles, ylabels):
        ax.set_title(t, fontsize=13)
        ax.set_xlabel("Epoch"); ax.set_ylabel(yl)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    path = os.path.join(OUTDIR, "training_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  saved → {path}")


def plot_size_vs_ppl(all_models: dict, all_histories: dict) -> None:
    names  = list(all_models.keys())
    params = [all_models[n].num_params() for n in names]
    ppls   = [all_histories[n]["val_ppl"][-1] for n in names]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(params, ppls, s=120, c=COLORS[:len(names)],
               zorder=5, edgecolors="white", linewidths=1)
    for x, y, label in zip(params, ppls, names):
        ax.annotate(label, (x, y), xytext=(8, 4),
                    textcoords="offset points", fontsize=9)

    ax.set_xscale("log")
    ax.set_xlabel("Parameters (log scale)", fontsize=12)
    ax.set_ylabel("Validation Perplexity", fontsize=12)
    ax.set_title("Model Size vs. Reasoning Ability\n(lower PPL = better)", fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    path = os.path.join(OUTDIR, "model_size_vs_perplexity.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  saved → {path}")


def plot_lr_schedules(all_histories: dict) -> None:
    fig, ax = plt.subplots(figsize=(9, 4))
    for i, (name, hist) in enumerate(all_histories.items()):
        ax.plot(range(1, len(hist["lr"]) + 1), hist["lr"],
                color=COLORS[i % len(COLORS)], label=name, lw=2)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Learning Rate")
    ax.set_title("LR Schedule (Cosine Annealing)", fontsize=13)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    path = os.path.join(OUTDIR, "lr_schedule.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  saved → {path}")


def plot_attention(model: MiniGPT, tokenizer: Tokenizer,
                   text: str, layer: int = 0) -> None:
    model.eval()
    ids = tokenizer.encode(text)[:20]
    x   = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        _ = model(x)

    attn = model.blocks[layer].attn.last_attn_weights   # (1, h, T, T)
    T    = attn.size(-1)
    labels = [tokenizer.id_to_token(i)[:6] for i in ids[:T]]
    n_show = min(attn.size(1), 4)

    fig, axes = plt.subplots(1, n_show, figsize=(4 * n_show, 4))
    if n_show == 1: axes = [axes]

    for h, ax in enumerate(axes):
        mat = attn[0, h, :T, :T].cpu().numpy()
        im  = ax.imshow(mat, cmap="Blues", vmin=0, vmax=mat.max())
        ax.set_title(f"Head {h+1}", fontsize=11)
        ax.set_xticks(range(T)); ax.set_xticklabels(labels, rotation=90, fontsize=7)
        ax.set_yticks(range(T)); ax.set_yticklabels(labels, fontsize=7)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle(f"Attention Weights — Layer {layer+1}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(OUTDIR, "attention_heatmap.png")
    plt.savefig(path, dpi=120, bbox_inches="tight"); plt.close()
    print(f"  saved → {path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"Device: {DEVICE}\n")

    tok_cfg = TokenizerConfig()
    print("Device:", DEVICE)
    print("Before fetch")
    text    = fetch_shakespeare()
    print("After fetch")
    tok     = build_tokenizer(text, tok_cfg)

    all_histories: dict = {}
    all_models:    dict = {}

    for exp in ABLATION_GRID:
        print(f"\n{'='*60}")
        print(f"  Experiment: {exp.name}")
        print(f"{'='*60}")

        train_dl, val_dl = build_dataloaders(text, tok, tok_cfg, exp.train)

        model = MiniGPT(
            vocab_size=len(tok),
            d_model=exp.model.d_model,
            n_layers=exp.model.n_layers,
            n_heads=exp.model.n_heads,
            d_ff=exp.model.d_ff,
            max_len=256,
            dropout=exp.model.dropout,
        ).to(DEVICE)
        model.summary()

        print("Before train")
        history = train(model, train_dl, val_dl, exp.train, DEVICE, tag=exp.name)
        print("After train")
        all_histories[exp.name] = history
        all_models[exp.name]    = model

    # ── plots ─────────────────────────────────────────────────────────────────
    print("\n[plots] generating...")
    plot_training_curves(all_histories)
    plot_size_vs_ppl(all_models, all_histories)
    plot_lr_schedules(all_histories)

    # use the medium model for attention heatmap
    if "medium_d256_h8" in all_models:
        plot_attention(
            all_models["medium_d256_h8"], tok,
            "to be or not to be that is the question"
        )

    # ── text samples ──────────────────────────────────────────────────────────
    print("\n[generation] samples from each model:")
    for name, model in all_models.items():
        sample = generate_sample(model, tok, prompt="HAMLET:")
        print(f"\n[ {name} ]\n{sample[:250]}\n")

    # ── save JSON results ─────────────────────────────────────────────────────
    results = {
        name: {
            "params":         all_models[name].num_params(),
            "final_val_loss": hist["val_loss"][-1],
            "final_val_ppl":  hist["val_ppl"][-1],
        }
        for name, hist in all_histories.items()
    }
    json_path = os.path.join(OUTDIR, "ablation_results.json")
    with open(json_path, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\n[results] saved → {json_path}")

    # ── print table ───────────────────────────────────────────────────────────
    print(f"\n{'Model':<22} {'Params':>10} {'Val Loss':>10} {'Val PPL':>9}")
    print("-" * 55)
    for name, r in results.items():
        print(f"{name:<22} {r['params']:>10,} {r['final_val_loss']:>10.4f} {r['final_val_ppl']:>9.1f}")


if __name__ == "__main__":
    main()
