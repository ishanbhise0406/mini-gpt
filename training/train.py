"""
training/train.py
─────────────────────────────────────────────────────────────────────────────
Training loop for MiniGPT.

Features
────────
  • AdamW or SGD optimiser (from config)
  • Cosine-annealing LR schedule
  • Gradient clipping
  • Per-epoch validation + perplexity
  • Checkpoint saving (best val loss)
  • History dict returned for plotting

Run directly
────────────
  python -m training.train
─────────────────────────────────────────────────────────────────────────────
"""

import math
import os
import time
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from model.transformer import MiniGPT
from training.config import ModelConfig, TrainConfig, TokenizerConfig
from training.dataset import fetch_shakespeare, build_tokenizer, build_dataloaders


# ── helpers ───────────────────────────────────────────────────────────────────

def perplexity(loss: float) -> float:
    return math.exp(min(loss, 20.0))


@torch.no_grad()
def evaluate(
    model:       MiniGPT,
    loader:      DataLoader,
    device:      torch.device,
    max_batches: int = 40,
) -> float:
    """Compute mean cross-entropy loss over `max_batches` validation batches."""
    model.eval()
    total, count = 0.0, 0
    for i, (x, y) in enumerate(loader):
        if i >= max_batches:
            break
        x, y    = x.to(device), y.to(device)
        logits  = model(x)
        B, T, V = logits.shape
        loss    = F.cross_entropy(logits.view(B * T, V), y.view(B * T))
        total  += loss.item()
        count  += 1
    return total / count if count else float("inf")


# ── main training function ────────────────────────────────────────────────────

def train(
    model:        MiniGPT,
    train_loader: DataLoader,
    val_loader:   DataLoader,
    train_cfg:    TrainConfig,
    device:       torch.device,
    tag:          str = "model",
) -> Dict:
    """
    Train `model` and return a history dictionary.

    Returns
    -------
    {
      "train_loss": [...],
      "val_loss":   [...],
      "train_ppl":  [...],
      "val_ppl":    [...],
      "lr":         [...],
    }
    """

    # ── optimiser ─────────────────────────────────────────────────────────────
    if train_cfg.optimizer == "adamw":
        optimizer = AdamW(
            model.parameters(),
            lr=train_cfg.lr,
            weight_decay=train_cfg.weight_decay,
        )
    else:
        optimizer = SGD(
            model.parameters(),
            lr=train_cfg.lr,
            momentum=0.9,
        )

    scheduler = (
        CosineAnnealingLR(optimizer, T_max=train_cfg.n_epochs)
        if train_cfg.scheduler == "cosine"
        else None
    )

    os.makedirs(train_cfg.save_dir, exist_ok=True)
    best_val   = float("inf")
    history    = {"train_loss": [], "val_loss": [],
                  "train_ppl":  [], "val_ppl":  [], "lr": []}
    t0 = time.time()

    for epoch in range(1, train_cfg.n_epochs + 1):
        # ── training epoch ────────────────────────────────────────────────────
        model.train()
        ep_loss, ep_steps = 0.0, 0

        for batch_idx, (x, y) in enumerate(train_loader):
            if batch_idx % 100 == 0:
                print(f"Batch {batch_idx}/{len(train_loader)}")
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)

            logits  = model(x)
            B, T, V = logits.shape
            loss    = F.cross_entropy(logits.view(B * T, V), y.view(B * T))
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), train_cfg.clip_grad)
            optimizer.step()

            ep_loss  += loss.item()
            ep_steps += 1

        if scheduler:
            scheduler.step()

        # ── validation ────────────────────────────────────────────────────────
        avg_train = ep_loss / ep_steps
        avg_val   = evaluate(model, val_loader, device)
        cur_lr    = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(avg_train)
        history["val_loss"].append(avg_val)
        history["train_ppl"].append(perplexity(avg_train))
        history["val_ppl"].append(perplexity(avg_val))
        history["lr"].append(cur_lr)

        if epoch % train_cfg.log_every == 0:
            elapsed = time.time() - t0
            print(
                f"[{tag}] epoch {epoch:>3}/{train_cfg.n_epochs} | "
                f"train={avg_train:.4f} ppl={perplexity(avg_train):.1f} | "
                f"val={avg_val:.4f} ppl={perplexity(avg_val):.1f} | "
                f"lr={cur_lr:.2e} | t={elapsed:.0f}s"
            )

        # ── checkpoint ────────────────────────────────────────────────────────
        if avg_val < best_val:
            best_val = avg_val
            ckpt = os.path.join(train_cfg.save_dir, f"{tag}_best.pt")
            torch.save(model.state_dict(), ckpt)

    print(f"[{tag}] best val loss = {best_val:.4f}")
    return history


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import random
    import numpy as np

    SEED   = 42
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

    tok_cfg   = TokenizerConfig()
    train_cfg = TrainConfig()
    model_cfg = ModelConfig()

    print(f"Device: {DEVICE}")

    text      = fetch_shakespeare(train_cfg.data_path)
    tokenizer = build_tokenizer(text, tok_cfg)
    train_dl, val_dl = build_dataloaders(text, tokenizer, tok_cfg, train_cfg)

    model = MiniGPT(
        vocab_size=len(tokenizer),
        d_model=model_cfg.d_model,
        n_layers=model_cfg.n_layers,
        n_heads=model_cfg.n_heads,
        d_ff=model_cfg.d_ff,
        max_len=tok_cfg.seq_len + 4,
        dropout=model_cfg.dropout,
    ).to(DEVICE)

    model.summary()
    history = train(model, train_dl, val_dl, train_cfg, DEVICE, tag="baseline")
    print("Training complete.")
