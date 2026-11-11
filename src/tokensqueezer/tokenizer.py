"""Byte-level BPE tokenizer (GPT-2 style), trained offline on a bundled corpus.

The tokenizer is a genuine byte-pair encoding implementation:

* text is pre-split with a GPT-2-style regex into words, numbers, punctuation
  runs and whitespace runs;
* each pre-token is encoded as UTF-8 bytes and then merged with a merge table
  learned from the bundled corpus (``corpus.txt``) — see ``train_bpe``;
* decoding is an exact round trip: tokens are byte sequences, so
  ``decode(encode(text)) == text`` for any valid Unicode string.

No network, no API key, fully deterministic. The merge table is shipped as
``merges.json`` so the runtime never needs to train; ``train_bpe`` is kept
public (and tested) so anyone can retrain on their own corpus.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

# GPT-2-style pre-tokenizer pattern (ASCII; other scripts fall into the
# punctuation class, which still round-trips exactly).
PATTERN = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+"""
)

DEFAULT_NUM_MERGES = 1000


def train_bpe(corpus: str, num_merges: int = DEFAULT_NUM_MERGES) -> list[tuple[bytes, bytes]]:
    """Learn BPE merges from *corpus*.

    Deterministic: ties between equally frequent pairs are broken by the
    lexicographic order of the pair, so the same corpus always produces the
    same merge table. Returns the merges in rank order (earliest first).
    """
    words: list[list[bytes]] = []
    for piece in PATTERN.findall(corpus):
        raw = piece.encode("utf-8")
        words.append([raw[i : i + 1] for i in range(len(raw))])

    merges: list[tuple[bytes, bytes]] = []
    for _ in range(num_merges):
        counts: dict[tuple[bytes, bytes], int] = {}
        for word in words:
            for a, b in zip(word, word[1:]):
                counts[(a, b)] = counts.get((a, b), 0) + 1
        if not counts:
            break
        # Most frequent pair; `sorted` makes tie-breaking deterministic.
        best = max(sorted(counts), key=lambda pair: counts[pair])
        merges.append(best)
        merged = best[0] + best[1]
        new_words: list[list[bytes]] = []
        for word in words:
            new_word: list[bytes] = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and (word[i], word[i + 1]) == best:
                    new_word.append(merged)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            words = new_words
    return merges


def merges_to_json(merges: list[tuple[bytes, bytes]]) -> str:
    """Serialize merges as JSON (latin-1 keeps the byte mapping lossless)."""
    return json.dumps(
        {"version": 1, "merges": [[a.decode("latin-1"), b.decode("latin-1")] for a, b in merges]},
        ensure_ascii=True,
    )


def merges_from_json(text: str) -> list[tuple[bytes, bytes]]:
    data = json.loads(text)
    return [(a.encode("latin-1"), b.encode("latin-1")) for a, b in data["merges"]]


class BPETokenizer:
    """Deterministic byte-level BPE tokenizer backed by a merge table."""

    def __init__(self, merges: list[tuple[bytes, bytes]]) -> None:
        self.merges = list(merges)
        self.ranks = {pair: rank for rank, pair in enumerate(self.merges)}
        self.vocab: dict[bytes, int] = {}
        for i in range(256):
            self.vocab[bytes([i])] = i
        for rank, (a, b) in enumerate(self.merges):
            self.vocab[a + b] = 256 + rank
        self.id_to_token = {token_id: token for token, token_id in self.vocab.items()}

    @classmethod
    def from_merges_file(cls, path: str | Path) -> "BPETokenizer":
        return cls(merges_from_json(Path(path).read_text(encoding="utf-8")))

    @classmethod
    def default(cls) -> "BPETokenizer":
        return cls.from_merges_file(Path(__file__).with_name("merges.json"))

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def _bpe(self, piece: bytes) -> list[bytes]:
        if len(piece) <= 1:
            return [piece[i : i + 1] for i in range(len(piece))]
        word = [piece[i : i + 1] for i in range(len(piece))]
        while len(word) > 1:
            best: tuple[bytes, bytes] | None = None
            best_rank: int | None = None
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                rank = self.ranks.get(pair)
                if rank is not None and (best_rank is None or rank < best_rank):
                    best, best_rank = pair, rank
            if best is None:
                break
            merged = best[0] + best[1]
            new_word: list[bytes] = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and (word[i], word[i + 1]) == best:
                    new_word.append(merged)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = new_word
        return word

    def encode(self, text: str) -> list[int]:
        """Encode *text* to token ids."""
        ids: list[int] = []
        for piece in PATTERN.findall(text):
            for token in self._bpe(piece.encode("utf-8")):
                ids.append(self.vocab[token])
        return ids

    def decode(self, ids: list[int]) -> str:
        """Decode token ids back to text (exact round trip for encoded text)."""
        return b"".join(self.id_to_token[i] for i in ids).decode("utf-8")

    def count(self, text: str) -> int:
        return len(self.encode(text))
