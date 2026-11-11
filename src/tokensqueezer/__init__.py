"""Token Squeezer — do more with fewer tokens.

Offline prompt compression, token counting and budget planning.
"""

from .benchmark import run_benchmark
from .budget import BudgetPlan, PromptPart, plan
from .compress import CompressionResult, compress
from .counter import count_tokens, estimate_cost, summarize
from .tokenizer import BPETokenizer

__version__ = "1.0.0"
__all__ = [
    "BPETokenizer",
    "CompressionResult",
    "BudgetPlan",
    "PromptPart",
    "compress",
    "count_tokens",
    "estimate_cost",
    "plan",
    "run_benchmark",
    "summarize",
]
