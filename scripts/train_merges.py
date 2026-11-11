"""Regenerate the shipped BPE merge table from the bundled corpus.

Usage:  python -m scripts.train_merges [--merges N]

Writes src/tokensqueezer/merges.json. Deterministic: running it twice
produces byte-identical output.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from tokensqueezer.tokenizer import DEFAULT_NUM_MERGES, merges_to_json, train_bpe

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "src" / "tokensqueezer"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train BPE merges on the bundled corpus.")
    parser.add_argument("--merges", type=int, default=DEFAULT_NUM_MERGES)
    args = parser.parse_args()

    corpus = (SRC / "corpus.txt").read_text(encoding="utf-8")
    merges = train_bpe(corpus, num_merges=args.merges)
    out = SRC / "merges.json"
    out.write_text(merges_to_json(merges), encoding="utf-8")
    print(f"trained {len(merges)} merges from {len(corpus)} chars -> {out}")


if __name__ == "__main__":
    main()
