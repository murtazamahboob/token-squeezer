"""Offline token counting and cost estimation.

Everything here is deterministic and needs no network: the count comes from
the shipped BPE tokenizer (``tokensqueezer.tokenizer``), and the cost estimate
is a plain multiplication you can price with any per-million-token rate.
"""

from __future__ import annotations

from functools import lru_cache

from .tokenizer import BPETokenizer


@lru_cache(maxsize=1)
def default_tokenizer() -> BPETokenizer:
    return BPETokenizer.default()


def count_tokens(text: str, tokenizer: BPETokenizer | None = None) -> int:
    """Return the number of BPE tokens in *text* (0 for empty text)."""
    return (tokenizer or default_tokenizer()).count(text)


def count_words(text: str) -> int:
    return len(text.split())


def estimate_cost(tokens: int, price_per_million: float = 2.50) -> float:
    """Estimated USD cost for *tokens* at *price_per_million* dollars."""
    return tokens * price_per_million / 1_000_000


def summarize(text: str, tokenizer: BPETokenizer | None = None) -> dict:
    """Token/word/character summary plus a cost estimate for *text*."""
    tok = tokenizer or default_tokenizer()
    tokens = tok.count(text)
    chars = len(text)
    return {
        "tokens": tokens,
        "words": count_words(text),
        "chars": chars,
        "chars_per_token": round(chars / tokens, 2) if tokens else 0.0,
        "estimated_cost_usd": round(estimate_cost(tokens), 6),
    }
