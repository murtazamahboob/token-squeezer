"""Deterministic demo: compress, plan a budget, benchmark — and write a report.

Runs fully offline with no API key. Writes two artifacts:
    outputs/report.md    (benchmark + worked examples, human-readable)
    outputs/summary.json (the same numbers, machine-readable)

Run:  python examples/demo.py        (or: tokensqueezer benchmark)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from tokensqueezer.benchmark import run_benchmark, write_report
from tokensqueezer.budget import PromptPart, plan
from tokensqueezer.compress import compress
from tokensqueezer.counter import default_tokenizer

WORKED_EXAMPLE = (
    "I would really like you to help me with a very important task. In order to get started, "
    "please read the following document very carefully. The document is actually a long report "
    "about artificial intelligence and machine learning, and it is basically about how large "
    "language models work. Due to the fact that the report is very long, I need a summary. "
    "Please summarize the report in three bullet points. The summary should be in plain language "
    "and it should be suitable for a non-expert reader. The summary should be in plain language."
)

BUDGET_PARTS = [
    PromptPart(
        name="system",
        text="You are a helpful assistant. Answer concisely and always cite your sources.",
        priority=1,
    ),
    PromptPart(
        name="context",
        text=(
            "The quarterly report shows revenue grew 12 percent. The growth was actually driven "
            "by the new pricing plan and by strong demand in Asia. The new pricing plan increased "
            "average revenue per user. The new pricing plan raised average revenue per user. "
            "Demand in Asia was strong throughout the quarter. The report also notes that costs "
            "rose 4 percent due to hiring."
        ),
        priority=3,
    ),
    PromptPart(
        name="examples",
        text=(
            "Example: Q: What drove growth? A: The new pricing plan and strong demand in Asia. "
            "Example: Q: Did costs rise? A: Yes, costs rose 4 percent due to hiring."
        ),
        priority=2,
    ),
    PromptPart(
        name="question",
        text="What drove revenue growth this quarter, and by how much did revenue grow?",
        priority=1,
    ),
]


def main() -> None:
    tok = default_tokenizer()
    out = ROOT / "outputs"
    out.mkdir(exist_ok=True)

    # 1. Worked example: one verbose prompt, per-stage savings.
    result = compress(WORKED_EXAMPLE, tokenizer=tok)
    print("== worked example ==")
    print(f"tokens: {result.tokens_before} -> {result.tokens_after} "
          f"({result.tokens_saved} saved, {result.pct_saved}%)")
    for stage in result.stages:
        print(f"  {stage.name}: {stage.tokens_before} -> {stage.tokens_after} ({stage.detail})")

    # 2. Budget planner: same 4 parts at two budgets — tight (drops) and
    # roomy (compression alone is enough).
    budget_results = {}
    for budget in (120, 280):
        budget_result = plan(BUDGET_PARTS, budget=budget, tokenizer=tok)
        budget_results[budget] = budget_result
        print(f"\n== budget plan ({budget} tokens) ==")
        for d in budget_result.parts:
            print(f"  {d.name}: {d.action} ({d.tokens_before} -> {d.tokens_after})")
        print(f"  total: {budget_result.tokens_used}/{budget} ({budget_result.utilization_pct}%), "
              f"saved {budget_result.tokens_saved}")

    # 3. Benchmark over the bundled samples.
    bench = run_benchmark(tokenizer=tok)
    print("\n== benchmark ==")
    print(f"{len(bench.samples)} samples: {bench.total_before} -> {bench.total_after} tokens "
          f"({bench.total_saved} saved, {bench.mean_pct_saved}% mean saving)")
    print(f"mean keyword retention: {bench.mean_keyword_retention}")
    print(f"mean fact retention: {bench.mean_fact_retention}")
    print(f"mean content-word overlap: {bench.mean_word_overlap}")

    # 4. Write artifacts: benchmark report + extra sections, and JSON summary.
    write_report(bench, out / "report.md")
    with (out / "report.md").open("a", encoding="utf-8") as fh:
        fh.write("\n## Worked example (per-stage savings)\n\n")
        fh.write(f"Original: {result.tokens_before} tokens, compressed: {result.tokens_after} "
                 f"tokens ({result.pct_saved}% saved).\n\n")
        fh.write("| Stage | Before | After | Detail |\n|---|---:|---:|---|\n")
        for stage in result.stages:
            fh.write(f"| {stage.name} | {stage.tokens_before} | {stage.tokens_after} | {stage.detail} |\n")
        fh.write(f"\nCompressed text:\n\n{result.compressed}\n")
        fh.write("\n## Budget plan examples\n\n")
        for budget, budget_result in budget_results.items():
            fh.write(f"### {budget}-token budget\n\n")
            fh.write("| Part | Action | Before | After |\n|---|---:|---:|---:|\n")
            for d in budget_result.parts:
                fh.write(f"| {d.name} | {d.action} | {d.tokens_before} | {d.tokens_after} |\n")
            fh.write(f"\nTotal: {budget_result.tokens_used}/{budget} tokens "
                     f"({budget_result.utilization_pct}%), saved {budget_result.tokens_saved}.\n\n")

    from dataclasses import asdict as _asdict

    summary = {
        "worked_example": {
            "tokens_before": result.tokens_before,
            "tokens_after": result.tokens_after,
            "tokens_saved": result.tokens_saved,
            "pct_saved": result.pct_saved,
            "stages": [
                {"name": s.name, "before": s.tokens_before, "after": s.tokens_after, "detail": s.detail}
                for s in result.stages
            ],
        },
        "budget_plans": [
            {
                "budget": budget,
                "tokens_used": budget_result.tokens_used,
                "tokens_saved": budget_result.tokens_saved,
                "utilization_pct": budget_result.utilization_pct,
                "parts": [
                    {"name": d.name, "action": d.action, "before": d.tokens_before, "after": d.tokens_after}
                    for d in budget_result.parts
                ],
            }
            for budget, budget_result in budget_results.items()
        ],
        "benchmark": {
            "total_before": bench.total_before,
            "total_after": bench.total_after,
            "total_saved": bench.total_saved,
            "mean_pct_saved": bench.mean_pct_saved,
            "mean_keyword_retention": bench.mean_keyword_retention,
            "mean_fact_retention": bench.mean_fact_retention,
            "mean_word_overlap": bench.mean_word_overlap,
            "samples": [_asdict(s) for s in bench.samples],
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\nreport: {out / 'report.md'}")
    print(f"summary: {out / 'summary.json'}")


if __name__ == "__main__":
    main()
