"""
model/transformer.py
─────────────────────────────────────────────────────────────────────────────
Full GPT-style decoder-only Transformer.

Block structure (Pre-LayerNorm):
  x → LN → MultiHeadAttention → residual add
    → LN → FeedForward         → residual add

Full model:
  token ids → Embedding (scaled) → SinusoidalPE
            → N × TransformerBlock
            → LayerNorm → Linear head (→ logits)
─────────────────────────────────────────────────────────────────────────────
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.attention import MultiHeadAttention
from model.positional import SinusoidalPositionalEncoding


# ── Feed-Forward Network ──────────────────────────────────────────────────────

class FeedForward(nn.Module):
    """
    Position-wise two-layer MLP with GELU activation (GPT-2 style).

    Architecture: Linear(d_model → d_ff) → GELU → Dropout
                → Linear(d_ff   → d_model) → Dropout
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── Transformer Block ─────────────────────────────────────────────────────────

class TransformerBlock(nn.Module):
    """
    One Pre-LN Transformer decoder block.

    Pre-LayerNorm (normalise before sub-layer, not after) is more stable
    during training than the original post-LN formulation.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.ln1  = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ln2  = nn.LayerNorm(d_model)
        self.ff   = FeedForward(d_model, d_ff, dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Self-attention sub-layer
        x = x + self.attn(self.ln1(x), mask)
        # Feed-forward sub-layer
        x = x + self.ff(self.ln2(x))
        return x


# ── MiniGPT ───────────────────────────────────────────────────────────────────

class MiniGPT(nn.Module):
    """
    Decoder-only Transformer language model (GPT-style).

    Parameters
    ----------
    vocab_size : number of tokens in the vocabulary
    d_model    : embedding / hidden dimension
    n_layers   : number of stacked TransformerBlocks
    n_heads    : attention heads per block
    d_ff       : hidden dimension of the feed-forward layer
    max_len    : maximum supported sequence length
    dropout    : dropout probability throughout
    """

    def __init__(
        self,
        vocab_size: int,
        d_model:    int = 128,
        n_layers:   int = 4,
        n_heads:    int = 4,
        d_ff:       int = 512,
        max_len:    int = 256,
        dropout:    float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model    = d_model
        self.n_layers   = n_layers
        self.n_heads    = n_heads

        # ── sub-modules ───────────────────────────────────────────────────────
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len, dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        self.ln_final = nn.LayerNorm(d_model)
        self.lm_head  = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying — output projection shares weights with token embedding.
        # Reduces parameters and improves performance (Press & Wolf, 2017).
        self.lm_head.weight = self.tok_emb.weight

        self._init_weights()

    # ── initialisation ────────────────────────────────────────────────────────

    def _init_weights(self) -> None:
        """GPT-2-style weight initialisation."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # ── causal mask ───────────────────────────────────────────────────────────

    @staticmethod
    def _causal_mask(T: int, device: torch.device) -> torch.Tensor:
        """Lower-triangular mask: token i can only attend to positions ≤ i."""
        return torch.tril(torch.ones(T, T, device=device)).view(1, 1, T, T)

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        idx : (B, T) — integer token ids

        Returns
        -------
        logits : (B, T, vocab_size)
        """
        B, T = idx.shape
        mask = self._causal_mask(T, idx.device)

        # Embed tokens and add positional signal
        x = self.tok_emb(idx) * math.sqrt(self.d_model)   # scale as in paper
        x = self.pos_enc(x)

        # Pass through each transformer block
        for block in self.blocks:
            x = block(x, mask)

        x = self.ln_final(x)
        return self.lm_head(x)

    # ── generation ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int = 200,
        temperature: float = 1.0,
        top_k: int = 40,
    ) -> torch.Tensor:
        """
        Autoregressive text generation with temperature scaling + top-k sampling.

        Parameters
        ----------
        idx            : (1, T) seed token ids
        max_new_tokens : how many new tokens to sample
        temperature    : > 1 → more random, < 1 → more deterministic
        top_k          : restrict sampling to the top-k most likely tokens

        Returns
        -------
        (1, T + max_new_tokens) token ids
        """
        self.eval()
        for _ in range(max_new_tokens):
            logits = self(idx)[:, -1, :] / temperature           # (1, V)

            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs     = F.softmax(logits, dim=-1)
            next_tok  = torch.multinomial(probs, num_samples=1)  # (1, 1)
            idx       = torch.cat([idx, next_tok], dim=1)

        return idx

    # ── utilities ─────────────────────────────────────────────────────────────

    def num_params(self, trainable_only: bool = True) -> int:
        return sum(
            p.numel() for p in self.parameters()
            if (not trainable_only or p.requires_grad)
        )

    def summary(self) -> None:
        sep = "=" * 52
        print(sep)
        print("  MiniGPT — Architecture Summary")
        print(sep)
        print(f"  vocab_size : {self.vocab_size:>10,}")
        print(f"  d_model    : {self.d_model:>10}")
        print(f"  n_layers   : {self.n_layers:>10}")
        print(f"  n_heads    : {self.n_heads:>10}")
        print(f"  parameters : {self.num_params():>10,}")
        print(sep)
