"""Tests for the token budget planner."""

import pytest

from tokensqueezer.budget import PromptPart, plan
from tokensqueezer.counter import default_tokenizer

SYSTEM = "You are a helpful assistant. Answer concisely."
CONTEXT = (
    "The quarterly report shows revenue grew 12 percent. The growth was driven by the "
    "new pricing plan and by strong demand in Asia. The new pricing plan increased average "
    "revenue per user. Demand in Asia was strong throughout the quarter."
)
EXAMPLES = "Example: Q: What drove growth? A: The new pricing plan and strong demand in Asia."
QUESTION = "What drove revenue growth this quarter?"


def _parts():
    return [
        PromptPart(name="system", text=SYSTEM, priority=1),
        PromptPart(name="context", text=CONTEXT, priority=3),
        PromptPart(name="examples", text=EXAMPLES, priority=2),
        PromptPart(name="question", text=QUESTION, priority=1),
    ]


def test_fits_when_under_budget():
    result = plan(_parts(), budget=1000)
    assert all(d.action == "keep" for d in result.parts)
    assert result.tokens_saved == 0
    assert result.text  # assembled in order


def test_assembled_text_preserves_order():
    result = plan(_parts(), budget=1000)
    assert result.text.index("helpful assistant") < result.text.index("quarterly report")
    assert result.text.index("quarterly report") < result.text.index("Example:")
    assert result.text.index("Example:") < result.text.index("What drove revenue")


def test_compresses_low_priority_first():
    tok = default_tokenizer()
    total = sum(tok.count(p.text) for p in _parts())
    # Budget just below the total: compression should be enough, no drops.
    result = plan(_parts(), budget=total - 5)
    assert result.tokens_used <= result.budget
    by_name = {d.name: d for d in result.parts}
    assert by_name["system"].action == "keep"
    assert by_name["question"].action == "keep"
    assert by_name["context"].action in ("compress", "truncate", "drop")


def test_drops_lowest_priority_when_needed():
    result = plan(_parts(), budget=40)
    by_name = {d.name: d for d in result.parts}
    assert by_name["context"].action == "drop"
    assert result.tokens_used <= 40


def test_never_drops_priority_1():
    result = plan(_parts(), budget=10)
    by_name = {d.name: d for d in result.parts}
    assert by_name["system"].action != "drop"
    assert by_name["question"].action != "drop"
    assert result.tokens_used <= 10


def test_zero_budget_drops_everything():
    result = plan(_parts(), budget=0)
    assert result.text == ""
    assert result.tokens_used == 0
    assert all(d.action == "drop" for d in result.parts)
    assert result.warnings


def test_single_part_over_budget_is_truncated():
    part = [PromptPart(name="only", text=CONTEXT, priority=1)]
    result = plan(part, budget=10)
    assert result.tokens_used <= 10
    assert result.parts[0].action in ("truncate", "compress")


def test_empty_part_kept_at_zero_tokens():
    parts = [PromptPart(name="empty", text="", priority=1), PromptPart(name="q", text=QUESTION, priority=1)]
    result = plan(parts, budget=100)
    by_name = {d.name: d for d in result.parts}
    assert by_name["empty"].tokens_after == 0


def test_invalid_priority_raises():
    with pytest.raises(ValueError):
        PromptPart(name="bad", text="x", priority=5)


def test_utilization_pct():
    result = plan(_parts(), budget=1000)
    assert 0.0 < result.utilization_pct <= 100.0


def test_tokens_used_never_exceeds_budget():
    # Core contract: for every budget, the assembled prompt fits.
    for budget in [0, 1, 5, 10, 15, 20, 25, 30, 40, 60, 100, 1000]:
        result = plan(_parts(), budget=budget)
        assert result.tokens_used <= budget, f"budget={budget}, used={result.tokens_used}"
        tok = default_tokenizer()
        assert tok.count(result.text) == result.tokens_used
