"""
training/dataset.py
─────────────────────────────────────────────────────────────────────────────
Data loading, tokenization, and DataLoader construction.

Supports:
  • Tiny Shakespeare  (auto-downloaded)
  • Wikipedia subset  (place wiki_subset.txt in data/)
─────────────────────────────────────────────────────────────────────────────
"""

import os
import requests
from typing import List, Tuple

import torch
from torch.utils.data import Dataset, DataLoader

from tokenizer.bpe import BPETrainer
from tokenizer.encode_decode import Tokenizer
from training.config import TokenizerConfig, TrainConfig


# ── Text fetching ─────────────────────────────────────────────────────────────

SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/"
    "master/data/tinyshakespeare/input.txt"
)


def fetch_shakespeare(save_path: str = "data/shakespeare.txt") -> str:
    """Download Tiny Shakespeare if not already present."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if not os.path.exists(save_path):
        print(f"[dataset] Downloading Tiny Shakespeare...")
        resp = requests.get(SHAKESPEARE_URL, timeout=30)
        resp.raise_for_status()
        with open(save_path, "w", encoding="utf-8") as fh:
            fh.write(resp.text)
        print(f"[dataset] Saved {len(resp.text):,} chars → {save_path}")
    with open(save_path, "r", encoding="utf-8") as fh:
        return fh.read()


def load_text(path: str) -> str:
    """Load any plain-text file."""
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read()


# ── PyTorch Dataset ───────────────────────────────────────────────────────────

class TokenDataset(Dataset):
    """
    Sliding-window dataset over a flat token-id array.

    Each sample is a (input_ids, target_ids) pair of length `seq_len`,
    where target_ids = input_ids shifted right by one position.

    Example (seq_len=4):
      ids    = [5, 7, 2, 9, 3, 1]
      sample = ([5, 7, 2, 9], [7, 2, 9, 3])
    """

    def __init__(self, token_ids: List[int], seq_len: int):
        self.ids     = torch.tensor(token_ids, dtype=torch.long)
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.ids) - self.seq_len

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        chunk = self.ids[idx : idx + self.seq_len + 1]
        return chunk[:-1], chunk[1:]


# ── DataLoader builder ────────────────────────────────────────────────────────

def build_tokenizer(
    text: str,
    tok_cfg: TokenizerConfig,
    cache_path: str = "data/tokenizer.pkl",
) -> Tokenizer:
    """Train (or load cached) BPE tokenizer."""
    trainer = BPETrainer(vocab_size=tok_cfg.vocab_size)
    if os.path.exists(cache_path):
        trainer.load(cache_path)
    else:
        trainer.train(text[:300_000], verbose=True)
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        trainer.save(cache_path)
    return Tokenizer(trainer)


def build_dataloaders(
    text:        str,
    tokenizer:   Tokenizer,
    tok_cfg:     TokenizerConfig,
    train_cfg:   TrainConfig,
) -> Tuple[DataLoader, DataLoader]:
    """
    Tokenize text, split into train/val, return DataLoaders.

    Returns
    -------
    (train_loader, val_loader)
    """
    ids    = tokenizer.encode(text)
    split  = int(len(ids) * train_cfg.train_split)

    train_ds = TokenDataset(ids[:split],  tok_cfg.seq_len)
    val_ds   = TokenDataset(ids[split:],  tok_cfg.seq_len)

    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=train_cfg.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=0,
    )

    print(
        f"[dataset] train tokens={split:,}  val tokens={len(ids)-split:,}\n"
        f"          train batches={len(train_loader)}  val batches={len(val_loader)}"
    )
    return train_loader, val_loader
