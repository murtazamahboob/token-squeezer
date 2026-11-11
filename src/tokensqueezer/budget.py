"""Token budget planner: fit a multi-part prompt into a token budget.

A prompt is a list of parts (system message, retrieved context, few-shot
examples, user question), each with a priority: 1 = must keep, 2 = important,
3 = nice to have. The planner, in order:

1. keeps everything when it already fits;
2. compresses the lowest-priority parts first;
3. drops the lowest-priority parts entirely (never priority 1);
4. truncates what is left, lowest priority first, until the budget is met.

The result is a plan: per-part decisions, the assembled prompt, and the
tokens used/saved. Deterministic and offline.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .compress import compress, truncate_to_budget
from .counter import default_tokenizer
from .tokenizer import BPETokenizer


@dataclass
class PromptPart:
    name: str
    text: str
    priority: int = 2  # 1 = must keep, 2 = important, 3 = nice to have

    def __post_init__(self) -> None:
        if self.priority not in (1, 2, 3):
            raise ValueError("priority must be 1, 2 or 3")


@dataclass
class PartDecision:
    name: str
    action: str  # keep | compress | truncate | drop
    tokens_before: int
    tokens_after: int
    text: str


@dataclass
class BudgetPlan:
    parts: list[PartDecision]
    text: str
    tokens_used: int
    budget: int
    tokens_saved: int
    warnings: list[str] = field(default_factory=list)

    @property
    def utilization_pct(self) -> float:
        if not self.budget:
            return 0.0
        return round(100.0 * self.tokens_used / self.budget, 1)


def plan(parts: list[PromptPart], budget: int, tokenizer: BPETokenizer | None = None) -> BudgetPlan:
    """Fit *parts* into *budget* tokens; returns the plan."""
    tok = tokenizer or default_tokenizer()
    decisions: dict[str, PartDecision] = {}
    warnings: list[str] = []

    for part in parts:
        decisions[part.name] = PartDecision(
            name=part.name,
            action="keep",
            tokens_before=tok.count(part.text),
            tokens_after=tok.count(part.text),
            text=part.text,
        )

    def assembled() -> str:
        kept = [decisions[p.name] for p in parts if decisions[p.name].action != "drop"]
        return "\n\n".join(d.text for d in kept if d.text)

    def total() -> int:
        # Exact count of the assembled prompt, including the "\n\n" joins —
        # the sum of part counts alone would under-count and break the budget.
        return tok.count(assembled())

    if budget <= 0:
        for d in decisions.values():
            d.action = "drop"
            d.tokens_after = 0
            d.text = ""
        warnings.append("budget is zero or negative: everything dropped")
        return BudgetPlan(
            parts=[decisions[p.name] for p in parts],
            text="",
            tokens_used=0,
            budget=budget,
            tokens_saved=sum(d.tokens_before for d in decisions.values()),
            warnings=warnings,
        )

    if total() <= budget:
        return BudgetPlan(
            parts=[decisions[p.name] for p in parts],
            text=assembled(),
            tokens_used=total(),
            budget=budget,
            tokens_saved=0,
        )

    by_priority = sorted(parts, key=lambda p: (-p.priority, parts.index(p)))

    # Step 1: compress lowest-priority parts first.
    for part in by_priority:
        if total() <= budget:
            break
        result = compress(decisions[part.name].text, tokenizer=tok)
        if result.tokens_after < decisions[part.name].tokens_after:
            decisions[part.name].action = "compress"
            decisions[part.name].tokens_after = result.tokens_after
            decisions[part.name].text = result.compressed

    # Step 2: drop lowest-priority parts (never priority 1).
    for part in by_priority:
        if total() <= budget:
            break
        if part.priority == 1:
            continue
        decisions[part.name].action = "drop"
        decisions[part.name].tokens_after = 0
        decisions[part.name].text = ""

    # Step 3: truncate what is left until it fits. Lowest priority first;
    # within one priority, later parts first, so the opening system message
    # is the last thing to lose tokens.
    for part in sorted(parts, key=lambda p: (-p.priority, -parts.index(p))):
        if total() <= budget:
            break
        d = decisions[part.name]
        if d.action == "drop":
            continue
        allowed = budget - (total() - d.tokens_after)
        if allowed < 0:
            allowed = 0
        truncated, _, _ = truncate_to_budget(d.text, allowed, tok)
        if tok.count(truncated) < d.tokens_after or allowed == 0:
            d.action = "truncate" if d.text != truncated else d.action
            d.tokens_after = tok.count(truncated)
            d.text = truncated
        if part.priority == 1 and d.action == "truncate":
            warnings.append(f"must-keep part '{part.name}' was truncated to fit the budget")
        if part.priority == 1 and allowed == 0 and not d.text:
            warnings.append(f"must-keep part '{part.name}' was emptied to fit the budget")

    text = assembled()
    saved = sum(d.tokens_before - d.tokens_after for d in decisions.values())
    return BudgetPlan(
        parts=[decisions[p.name] for p in parts],
        text=text,
        tokens_used=tok.count(text),
        budget=budget,
        tokens_saved=saved,
        warnings=warnings,
    )
