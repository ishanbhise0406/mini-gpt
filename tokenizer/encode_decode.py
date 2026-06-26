"""
tokenizer/encode_decode.py
─────────────────────────────────────────────────────────────────────────────
Encoding (text → token ids) and decoding (token ids → text) using a trained
BPETrainer instance.
─────────────────────────────────────────────────────────────────────────────
"""

import re
from typing import List

from tokenizer.bpe import BPETrainer


class Tokenizer:
    """
    Wraps a trained BPETrainer and exposes clean encode / decode methods.

    Usage
    ─────
    tok = Tokenizer(trainer)
    ids = tok.encode("To be or not to be")
    txt = tok.decode(ids)
    """

    def __init__(self, trainer: BPETrainer):
        assert trainer._trained, "BPETrainer must be trained before wrapping."
        self.trainer = trainer
        self.vocab     = trainer.vocab
        self.inv_vocab = trainer.inv_vocab
        self.merges    = trainer.merges

        # special token ids
        self.pad_id = self.vocab["<PAD>"]
        self.unk_id = self.vocab["<UNK>"]
        self.bos_id = self.vocab["<BOS>"]
        self.eos_id = self.vocab["<EOS>"]

    # ── core helpers ──────────────────────────────────────────────────────────

    def _apply_merges(self, word: str) -> List[str]:
        """Apply learned BPE merges to a single lowercase word."""
        symbols = list(word) + ["</w>"]
        for (a, b), merged in self.merges.items():
            i = 0
            while i < len(symbols) - 1:
                if symbols[i] == a and symbols[i + 1] == b:
                    symbols[i] = merged
                    del symbols[i + 1]
                else:
                    i += 1
        return symbols

    # ── public API ────────────────────────────────────────────────────────────

    def encode(
        self,
        text: str,
        add_special_tokens: bool = False,
        max_length: int = None,
    ) -> List[int]:
        """
        Convert raw text to a list of integer token ids.

        Parameters
        ----------
        text               : input string
        add_special_tokens : prepend <BOS> and append <EOS>
        max_length         : if set, truncate (before special tokens)
        """
        text = text.lower()
        ids: List[int] = []

        for word in re.findall(r"\S+", text):
            for sym in self._apply_merges(word):
                ids.append(self.vocab.get(sym, self.unk_id))

        if max_length is not None:
            ids = ids[:max_length]

        if add_special_tokens:
            ids = [self.bos_id] + ids + [self.eos_id]

        return ids

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        """
        Convert a list of token ids back to a human-readable string.

        Parameters
        ----------
        ids           : list of integer token ids
        skip_special  : if True, drop <PAD>/<BOS>/<EOS>/<UNK> from output
        """
        special_ids = {self.pad_id, self.bos_id, self.eos_id}
        tokens: List[str] = []
        for i in ids:
            if skip_special and i in special_ids:
                continue
            tokens.append(self.inv_vocab.get(i, "<UNK>"))
        # </w> marks word boundaries → replace with space
        return "".join(tokens).replace("</w>", " ").strip()

    def encode_batch(self, texts: List[str], pad: bool = True,
                     max_length: int = 128) -> List[List[int]]:
        """Encode a list of strings, optionally padding to the same length."""
        encoded = [self.encode(t, max_length=max_length) for t in texts]
        if pad:
            target_len = max(len(e) for e in encoded)
            encoded = [
                e + [self.pad_id] * (target_len - len(e)) for e in encoded
            ]
        return encoded

    # ── utilities ─────────────────────────────────────────────────────────────

    def vocab_size(self) -> int:
        return len(self.vocab)

    def token_to_id(self, token: str) -> int:
        return self.vocab.get(token, self.unk_id)

    def id_to_token(self, idx: int) -> str:
        return self.inv_vocab.get(idx, "<UNK>")

    def __len__(self) -> int:
        return len(self.vocab)

    def __repr__(self) -> str:
        return f"Tokenizer(vocab_size={len(self.vocab)})"
