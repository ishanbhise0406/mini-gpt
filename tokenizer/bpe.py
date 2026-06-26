"""
tokenizer/bpe.py
─────────────────────────────────────────────────────────────────────────────
Byte-Pair Encoding (BPE) tokenizer — trained from scratch, no HuggingFace.

Algorithm (Sennrich et al., 2016):
  1. Build a character-level vocabulary from word frequencies.
  2. Count every adjacent symbol pair across the corpus.
  3. Merge the most frequent pair into a new token.
  4. Repeat until vocab_size is reached.
─────────────────────────────────────────────────────────────────────────────
"""

import re
import pickle
from collections import Counter
from typing import Dict, List, Tuple, Optional


SPECIAL_TOKENS: Dict[str, int] = {
    "<PAD>": 0,
    "<UNK>": 1,
    "<BOS>": 2,
    "<EOS>": 3,
}


# ── low-level helpers ─────────────────────────────────────────────────────────

def word_to_chars(word: str) -> Tuple[str, ...]:
    """'hello' → ('h','e','l','l','o','</w>')"""
    return tuple(list(word) + ["</w>"])


def get_pair_stats(vocab_freq: Dict[Tuple, int]) -> Counter:
    """Count every adjacent (a, b) pair weighted by word frequency."""
    pairs: Counter = Counter()
    for symbols, freq in vocab_freq.items():
        for i in range(len(symbols) - 1):
            pairs[(symbols[i], symbols[i + 1])] += freq
    return pairs


def merge_pair(
    pair: Tuple[str, str],
    vocab_freq: Dict[Tuple, int],
) -> Dict[Tuple, int]:
    """Replace every occurrence of `pair` with the merged token."""
    a, b = pair
    merged = a + b
    new_vocab: Dict[Tuple, int] = {}
    for symbols, freq in vocab_freq.items():
        new_syms: List[str] = []
        i = 0
        while i < len(symbols):
            if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
                new_syms.append(merged)
                i += 2
            else:
                new_syms.append(symbols[i])
                i += 1
        new_vocab[tuple(new_syms)] = freq
    return new_vocab


# ── BPETrainer ────────────────────────────────────────────────────────────────

class BPETrainer:
    """
    Learns BPE merge rules from a raw text corpus.

    Usage
    ─────
    trainer = BPETrainer(vocab_size=1500)
    trainer.train(text)
    trainer.save("tokenizer/bpe.pkl")
    """

    def __init__(self, vocab_size: int = 1500):
        self.vocab_size = vocab_size
        self.merges: Dict[Tuple[str, str], str] = {}   # (a,b) → merged
        self.vocab:  Dict[str, int] = {}
        self.inv_vocab: Dict[int, str] = {}
        self._trained = False

    # ── training ──────────────────────────────────────────────────────────────

    def train(self, text: str, verbose: bool = True) -> None:
        """Learn merge rules from raw text."""

        # 1. word frequencies
        words = re.findall(r"\S+", text.lower())
        word_freq = Counter(words)

        # 2. represent each word as a tuple of chars + </w>
        vocab_freq: Dict[Tuple, int] = {
            word_to_chars(w): f for w, f in word_freq.items()
        }

        # 3. collect base characters → initial vocab
        base_chars: set = set()
        for syms in vocab_freq:
            base_chars.update(syms)

        self.vocab = dict(SPECIAL_TOKENS)
        for ch in sorted(base_chars):
            if ch not in self.vocab:
                self.vocab[ch] = len(self.vocab)

        # 4. iterative merges
        n_merges = self.vocab_size - len(self.vocab)
        if verbose:
            print(f"[BPE] base vocab={len(self.vocab)} | target merges={n_merges}")

        for i in range(n_merges):
            pairs = get_pair_stats(vocab_freq)
            if not pairs:
                break

            best: Tuple[str, str] = max(pairs, key=pairs.get)
            new_token = best[0] + best[1]
            self.merges[best] = new_token
            vocab_freq = merge_pair(best, vocab_freq)

            if new_token not in self.vocab:
                self.vocab[new_token] = len(self.vocab)

            if verbose and (i + 1) % 200 == 0:
                print(
                    f"  merge {i+1:>5}/{n_merges} | "
                    f"vocab={len(self.vocab):>5} | "
                    f"merged={new_token!r:<20} freq={pairs[best]}"
                )

        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self._trained = True
        print(f"[BPE] training done. final vocab size: {len(self.vocab)}")

    # ── persistence ───────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        with open(path, "wb") as fh:
            pickle.dump(
                {"merges": self.merges, "vocab": self.vocab,
                 "vocab_size": self.vocab_size},
                fh,
            )
        print(f"[BPE] saved → {path}")

    def load(self, path: str) -> None:
        with open(path, "rb") as fh:
            data = pickle.load(fh)
        self.merges     = data["merges"]
        self.vocab      = data["vocab"]
        self.vocab_size = data.get("vocab_size", len(self.vocab))
        self.inv_vocab  = {v: k for k, v in self.vocab.items()}
        self._trained   = True
        print(f"[BPE] loaded from {path} | vocab={len(self.vocab)}")

    # ── helpers ───────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.vocab)

    def __repr__(self) -> str:
        status = f"vocab={len(self.vocab)}" if self._trained else "untrained"
        return f"BPETrainer({status})"
