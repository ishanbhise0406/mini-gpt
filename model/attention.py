"""
model/attention.py
─────────────────────────────────────────────────────────────────────────────
Self-Attention and Multi-Head Attention — built from scratch.

Scaled Dot-Product Attention
─────────────────────────────
  Attention(Q, K, V) = softmax( QKᵀ / √d_k ) · V

Multi-Head Attention
─────────────────────────────
  Split Q, K, V into h heads → run attention per head → concat → project.
─────────────────────────────────────────────────────────────────────────────
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Scaled Dot-Product Attention (functional) ─────────────────────────────────

def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    dropout: Optional[nn.Dropout] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Core attention operation.

    Parameters
    ----------
    Q, K, V : (..., seq_len, d_k)
    mask    : optional boolean mask — 0 positions are set to -inf
    dropout : optional dropout on attention weights

    Returns
    -------
    (context vectors, attention weights)  both shape (..., seq_len, d_k / d_v)
    """
    d_k = Q.size(-1)
    # ① raw scores
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)   # (..., T, T)

    # ② causal / padding mask
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))

    # ③ normalise → probabilities
    attn_weights = F.softmax(scores, dim=-1)

    # ④ optional dropout on weights
    if dropout is not None:
        attn_weights = dropout(attn_weights)

    # ⑤ weighted sum of values
    context = torch.matmul(attn_weights, V)                           # (..., T, d_v)
    return context, attn_weights


# ── Multi-Head Attention ───────────────────────────────────────────────────────

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Self-Attention.

    Projects Q, K, V into `n_heads` sub-spaces, applies scaled dot-product
    attention per head, concatenates the outputs, and projects back.

    Parameters
    ----------
    d_model  : total embedding dimension
    n_heads  : number of parallel attention heads
    dropout  : dropout probability on attention weights
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, (
            f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        )
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k     = d_model // n_heads          # dimension per head

        # Linear projections for Q, K, V (no bias — standard GPT practice)
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)

        # Output projection
        self.W_o = nn.Linear(d_model, d_model)

        self.attn_dropout = nn.Dropout(dropout)

        # Stored for external visualisation
        self.last_attn_weights: Optional[torch.Tensor] = None

    # ── helpers ───────────────────────────────────────────────────────────────

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, D) → (B, h, T, d_k)"""
        B, T, D = x.shape
        return (
            x.view(B, T, self.n_heads, self.d_k)
             .transpose(1, 2)
        )

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(B, h, T, d_k) → (B, T, D)"""
        B, h, T, d_k = x.shape
        return (
            x.transpose(1, 2)
             .contiguous()
             .view(B, T, self.d_model)
        )

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x    : (B, T, d_model)
        mask : (1, 1, T, T) causal mask — 0 = masked, 1 = allowed

        Returns
        -------
        (B, T, d_model)
        """
        Q = self._split_heads(self.W_q(x))   # (B, h, T, d_k)
        K = self._split_heads(self.W_k(x))
        V = self._split_heads(self.W_v(x))

        context, weights = scaled_dot_product_attention(
            Q, K, V, mask=mask, dropout=self.attn_dropout
        )
        self.last_attn_weights = weights.detach()   # store for viz

        out = self._merge_heads(context)             # (B, T, d_model)
        return self.W_o(out)
