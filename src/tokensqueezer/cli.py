"""Command-line interface for Token Squeezer.

Usage:
    tokensqueezer count "some text"            # token count + cost estimate
    tokensqueezer compress "some text"         # compressed text + stage stats
    tokensqueezer compress --budget 60 --json # JSON output, fits a budget
    tokensqueezer budget --budget 120 --parts parts.json
    tokensqueezer benchmark --out outputs/    # report.md + summary.json

Text can come from the argument or from stdin (pass ``-`` or pipe input).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .benchmark import run_benchmark, write_report, write_summary_json
from .budget import PromptPart, plan
from .compress import compress
from .counter import default_tokenizer, summarize


def _read_text(arg: str | None) -> str:
    if arg is None or arg == "-":
        return sys.stdin.read()
    return arg


def _cmd_count(args: argparse.Namespace) -> int:
    text = _read_text(args.text)
    info = summarize(text)
    print(f"tokens: {info['tokens']}")
    print(f"words: {info['words']}")
    print(f"chars: {info['chars']}")
    print(f"chars/token: {info['chars_per_token']}")
    print(f"est. cost @ $2.50/1M tokens: ${info['estimated_cost_usd']:.6f}")
    return 0


def _cmd_compress(args: argparse.Namespace) -> int:
    text = _read_text(args.text)
    result = compress(text, budget=args.budget)
    if args.json:
        print(
            json.dumps(
                {
                    "original": result.original,
                    "compressed": result.compressed,
                    "tokens_before": result.tokens_before,
                    "tokens_after": result.tokens_after,
                    "tokens_saved": result.tokens_saved,
                    "pct_saved": result.pct_saved,
                    "stages": [
                        {
                            "name": s.name,
                            "before": s.tokens_before,
                            "after": s.tokens_after,
                            "saved": s.saved,
                            "detail": s.detail,
                        }
                        for s in result.stages
                    ],
                },
                indent=2,
            )
        )
    else:
        print(result.compressed)
        print(
            f"\n[{result.tokens_before} -> {result.tokens_after} tokens, "
            f"saved {result.tokens_saved} ({result.pct_saved}%)]",
            file=sys.stderr,
        )
        for stage in result.stages:
            print(
                f"  {stage.name}: {stage.tokens_before} -> {stage.tokens_after} "
                f"({stage.detail})",
                file=sys.stderr,
            )
    return 0


def _cmd_budget(args: argparse.Namespace) -> int:
    parts_data = json.loads(Path(args.parts).read_text(encoding="utf-8"))
    parts = [
        PromptPart(name=p["name"], text=p["text"], priority=int(p.get("priority", 2)))
        for p in parts_data
    ]
    result = plan(parts, budget=args.budget)
    for d in result.parts:
        print(f"  {d.name}: {d.action} ({d.tokens_before} -> {d.tokens_after} tokens)")
    for warning in result.warnings:
        print(f"  warning: {warning}")
    print(f"total: {result.tokens_used}/{result.budget} tokens ({result.utilization_pct}%)")
    print(f"saved: {result.tokens_saved} tokens")
    print()
    print(result.text)
    return 0


def _cmd_benchmark(args: argparse.Namespace) -> int:
    result = run_benchmark()
    out = Path(args.out)
    write_report(result, out / "report.md")
    write_summary_json(result, out / "summary.json")
    print(
        f"{len(result.samples)} samples: {result.total_before} -> {result.total_after} "
        f"tokens ({result.total_saved} saved, {result.mean_pct_saved}% mean saving)"
    )
    print(f"mean keyword retention: {result.mean_keyword_retention}")
    print(f"mean fact retention: {result.mean_fact_retention}")
    print(f"report: {out / 'report.md'}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tokensqueezer",
        description="Do more with fewer tokens: compress prompts, plan token budgets, measure retention.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_count = sub.add_parser("count", help="Count tokens in text (arg or stdin).")
    p_count.add_argument("text", nargs="?", default=None)
    p_count.set_defaults(func=_cmd_count)

    p_compress = sub.add_parser("compress", help="Compress text to fewer tokens.")
    p_compress.add_argument("text", nargs="?", default=None)
    p_compress.add_argument("--budget", type=int, default=None, help="Max tokens for the result.")
    p_compress.add_argument("--json", action="store_true", help="Print a JSON result object.")
    p_compress.set_defaults(func=_cmd_compress)

    p_budget = sub.add_parser("budget", help="Fit prompt parts into a token budget.")
    p_budget.add_argument("--budget", type=int, required=True)
    p_budget.add_argument("--parts", required=True, help="JSON file with [{name, text, priority}].")
    p_budget.set_defaults(func=_cmd_budget)

    p_bench = sub.add_parser("benchmark", help="Run the retention benchmark and write a report.")
    p_bench.add_argument("--out", default="outputs")
    p_bench.set_defaults(func=_cmd_benchmark)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
