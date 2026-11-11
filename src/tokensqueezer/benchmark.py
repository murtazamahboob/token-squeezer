"""Benchmark: measure tokens saved vs meaning retained.

For every bundled sample (see ``data.py``) the compressor runs and three
retention metrics are measured on the result:

* **keyword retention** — fraction of the sample's essential keywords still
  present (case-insensitive whole-word/phrase match);
* **fact retention** — fraction of key facts whose content words are at
  least 60% present;
* **word overlap** — Jaccard similarity of the content-word sets.

The benchmark also writes a human-readable Markdown report and a JSON
summary, so the numbers are inspectable artifacts, not just console output.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .compress import apply_shorthand, compress, content_words, word_tokens
from .counter import default_tokenizer
from .data import SAMPLES
from .tokenizer import BPETokenizer

# Long form -> abbreviation, so retention metrics accept either spelling:
# "machine learning" compressed to "ML" still counts as retained.
SHORTHAND_MAP = {
    "artificial intelligence": "AI",
    "machine learning": "ML",
    "deep learning": "DL",
    "natural language processing": "NLP",
    "large language models": "LLMs",
    "large language model": "LLM",
    "application programming interface": "API",
    "user interface": "UI",
    "for example": "e.g.",
}


def _contains(phrase: str, text: str) -> bool:
    pattern = r"\b" + re.escape(phrase.lower()) + r"\b"
    return re.search(pattern, text.lower()) is not None


def keyword_retention(keywords: list[str], compressed: str) -> float:
    if not keywords:
        return 1.0
    kept = 0
    for kw in keywords:
        if _contains(kw, compressed):
            kept += 1
        elif kw.lower() in SHORTHAND_MAP and _contains(SHORTHAND_MAP[kw.lower()], compressed):
            kept += 1  # term survives as its abbreviation
    return round(kept / len(keywords), 3)


def fact_retention(facts: list[str], compressed: str) -> float:
    if not facts:
        return 1.0
    # Normalize facts to shorthand first, so "machine learning" in a fact
    # matches "ML" in the compressed text.
    present_words = set(word_tokens(apply_shorthand(compressed)[0].lower()))
    kept = 0
    for fact in facts:
        words = content_words(apply_shorthand(fact)[0])
        if not words:
            kept += 1
            continue
        hit = sum(1 for w in words if w in present_words)
        if hit / len(words) >= 0.6:
            kept += 1
    return round(kept / len(facts), 3)


def word_overlap(original: str, compressed: str) -> float:
    a = set(content_words(original))
    b = set(content_words(compressed))
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return round(len(a & b) / len(a | b), 3)


@dataclass
class SampleResult:
    id: str
    title: str
    tokens_before: int
    tokens_after: int
    tokens_saved: int
    pct_saved: float
    keyword_retention: float
    fact_retention: float
    word_overlap: float
    compressed: str


@dataclass
class BenchmarkResult:
    samples: list[SampleResult] = field(default_factory=list)

    @property
    def total_before(self) -> int:
        return sum(s.tokens_before for s in self.samples)

    @property
    def total_after(self) -> int:
        return sum(s.tokens_after for s in self.samples)

    @property
    def total_saved(self) -> int:
        return self.total_before - self.total_after

    @property
    def mean_pct_saved(self) -> float:
        if not self.samples:
            return 0.0
        return round(sum(s.pct_saved for s in self.samples) / len(self.samples), 1)

    @property
    def mean_keyword_retention(self) -> float:
        if not self.samples:
            return 0.0
        return round(sum(s.keyword_retention for s in self.samples) / len(self.samples), 3)

    @property
    def mean_fact_retention(self) -> float:
        if not self.samples:
            return 0.0
        return round(sum(s.fact_retention for s in self.samples) / len(self.samples), 3)

    @property
    def mean_word_overlap(self) -> float:
        if not self.samples:
            return 0.0
        return round(sum(s.word_overlap for s in self.samples) / len(self.samples), 3)


def run_benchmark(tokenizer: BPETokenizer | None = None) -> BenchmarkResult:
    tok = tokenizer or default_tokenizer()
    results: list[SampleResult] = []
    for sample in SAMPLES:
        result = compress(sample["text"], tokenizer=tok)
        results.append(
            SampleResult(
                id=sample["id"],
                title=sample["title"],
                tokens_before=result.tokens_before,
                tokens_after=result.tokens_after,
                tokens_saved=result.tokens_saved,
                pct_saved=result.pct_saved,
                keyword_retention=keyword_retention(sample["keywords"], result.compressed),
                fact_retention=fact_retention(sample["facts"], result.compressed),
                word_overlap=word_overlap(sample["text"], result.compressed),
                compressed=result.compressed,
            )
        )
    return BenchmarkResult(samples=results)


def write_report(result: BenchmarkResult, path: str | Path) -> Path:
    path = Path(path)
    lines = [
        "# Token Squeezer — Benchmark Report",
        "",
        "Compression pipeline: normalize -> fillers -> shorthand -> dedup. "
        "Token counts use the shipped offline BPE tokenizer.",
        "",
        "## Summary",
        "",
        f"- Samples: {len(result.samples)}",
        f"- Total tokens: {result.total_before} -> {result.total_after} "
        f"({result.total_saved} saved, {result.mean_pct_saved}% mean saving)",
        f"- Mean keyword retention: {result.mean_keyword_retention}",
        f"- Mean fact retention: {result.mean_fact_retention}",
        f"- Mean content-word overlap: {result.mean_word_overlap}",
        "",
        "## Per-sample results",
        "",
        "| Sample | Before | After | Saved | Keyword retention | Fact retention |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for s in result.samples:
        lines.append(
            f"| {s.title} | {s.tokens_before} | {s.tokens_after} | "
            f"{s.pct_saved}% | {s.keyword_retention} | {s.fact_retention} |"
        )
    lines.append("")
    lines.append("## Compressed outputs")
    lines.append("")
    for s in result.samples:
        lines.append(f"### {s.title} ({s.tokens_before} -> {s.tokens_after} tokens)")
        lines.append("")
        lines.append(s.compressed)
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_summary_json(result: BenchmarkResult, path: str | Path) -> Path:
    path = Path(path)
    data = {
        "total_before": result.total_before,
        "total_after": result.total_after,
        "total_saved": result.total_saved,
        "mean_pct_saved": result.mean_pct_saved,
        "mean_keyword_retention": result.mean_keyword_retention,
        "mean_fact_retention": result.mean_fact_retention,
        "mean_word_overlap": result.mean_word_overlap,
        "samples": [asdict(s) for s in result.samples],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path
