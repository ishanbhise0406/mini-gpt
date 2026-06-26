"""
training/config.py
─────────────────────────────────────────────────────────────────────────────
Central hyperparameter store.
Edit this file to change any setting — all other modules import from here.
─────────────────────────────────────────────────────────────────────────────
"""

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class TokenizerConfig:
    vocab_size: int   = 1500     # BPE vocabulary size
    seq_len:    int   = 128      # tokens per training sample


@dataclass
class ModelConfig:
    d_model:  int   = 128        # embedding / hidden dimension
    n_layers: int   = 4          # transformer blocks
    n_heads:  int   = 4          # attention heads (d_model % n_heads == 0)
    d_ff:     int   = 512        # feed-forward hidden dim (usually 4 × d_model)
    dropout:  float = 0.1        # dropout probability


@dataclass
class TrainConfig:
    # optimiser
    optimizer:    Literal["adamw", "sgd"] = "adamw"
    lr:           float = 3e-4
    weight_decay: float = 0.01
    # lr schedule
    scheduler:    Literal["cosine", "none"] = "cosine"
    # training
    n_epochs:     int   = 5
    batch_size:   int   = 16
    clip_grad:    float = 1.0
    train_split:  float = 0.9
    # logging
    log_every:    int   = 1      # log every N epochs
    save_dir:     str   = "checkpoints"
    # data
    data_path:    str   = "data/shakespeare.txt"


@dataclass
class ExperimentConfig:
    """One row in the ablation grid."""
    name:         str
    model:        ModelConfig
    train:        TrainConfig


# ── Experiment grid ──────────────────────────────────────────────────────────

ABLATION_GRID = [
    ExperimentConfig(
        name="tiny_d64_h2",
        model=ModelConfig(
            d_model=64,
            n_layers=2,
            n_heads=2,
            d_ff=256
        ),
        train=TrainConfig(
            n_epochs=10
        ),
    ),
]
