"""
model/positional.py
─────────────────────────────────────────────────────────────────────────────
Sinusoidal Positional Encoding — Vaswani et al. (2017).

PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))

Why fixed (not learned)?
  • Works well for sequences up to max_len without extra parameters.
  • Generalises to lengths not seen during training.
─────────────────────────────────────────────────────────────────────────────
"""

import math
import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    """
    Adds position-dependent patterns to token embeddings so the model
    knows the order of tokens.

    Parameters
    ----------
    d_model : embedding dimension
    max_len : maximum sequence length supported
    dropout : dropout applied after adding positional signal
    """

    def __init__(self, d_model: int, max_len: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Build the (max_len, d_model) table once and register as buffer
        pe = torch.zeros(max_len, d_model)                        # (L, D)
        position = torch.arange(max_len).unsqueeze(1).float()     # (L, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10_000.0) / d_model)
        )                                                          # (D/2,)

        pe[:, 0::2] = torch.sin(position * div_term)              # even dims
        pe[:, 1::2] = torch.cos(position * div_term)              # odd  dims

        # Shape: (1, max_len, d_model) — broadcast over batch
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (batch, seq_len, d_model)

        Returns
        -------
        x + positional encoding, same shape
        """
        x = x + self.pe[:, : x.size(1)]     # slice to actual seq length
        return self.dropout(x)
