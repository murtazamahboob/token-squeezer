# Token Squeezer — Do More With Fewer Tokens

**Compress prompts, plan token budgets, and measure what survives — fully offline, no API key, deterministic.**

Every request to a large language model is measured in tokens, and tokens cost money and time. Token Squeezer is a toolkit that shrinks prompts without shrinking their meaning: it removes filler, swaps wordy phrases for tight ones, replaces long terms with standard abbreviations, drops near-duplicate sentences, and fits multi-part prompts into a token budget — then proves the meaning survived with a measurable benchmark.

## Results (measured, not claimed)

Benchmark on 6 verbose real-world prompts (product copy, meeting notes, system prompts, research questions, code review, support context):

| Sample | Before | After | Saved |
|---|---:|---:|---:|
| Verbose product description | 237 | 186 | 21.5% |
| Rambling meeting notes | 321 | 244 | 24.0% |
| Overlong support-bot system prompt | 385 | 296 | 23.1% |
| Research question with repeated context | 310 | 267 | 13.9% |
| Verbose code review request | 369 | 328 | 11.1% |
| Customer context dump | 340 | 276 | 18.8% |
| **Total** | **1962** | **1597** | **365 tokens saved, 18.7% mean** |

Meaning retained: **keyword retention 1.0, key-fact retention 1.0, content-word overlap 0.889** — every essential keyword and key fact survived compression in all 6 samples.

Worked example: a verbose instruction compressed **150 → 118 tokens (21.3% saved)**. Budget planner: the same 4-part prompt fits a 280-token budget at **274/280 (97.9% utilization)** by compressing the context, and fits a 120-token budget by dropping the two lowest-priority parts while keeping the system message and the question untouched.

## Features

- **Real BPE tokenizer, built from scratch** — byte-level byte-pair encoding trained offline on a bundled corpus (1000 merges, 1256-token vocabulary). Exact encode/decode round trips, deterministic counts. Calibrated against `cl100k_base`: on a 2,000-character English passage it counts 437 tokens vs 423 (within ~3%).
- **5-stage compression pipeline** — normalize, filler removal, shorthand abbreviations, near-duplicate sentence removal (Jaccard + overlap coefficient), and value-based truncation to a token budget. Every stage reports its own token savings.
- **Budget planner** — split a prompt into prioritized parts (system, context, examples, question) and fit a token budget: compress low-priority parts first, drop them next (never priority 1), truncate last.
- **Retention benchmark** — keyword retention, key-fact retention, and content-word overlap, with a Markdown report and JSON summary as real artifacts.
- **CLI** — `count`, `compress`, `budget`, `benchmark` subcommands; reads text from an argument or stdin.

## Quickstart

```bash
pip install -r requirements.txt
pip install -e .

# Count tokens in a prompt
tokensqueezer count "In order to save tokens, remove very long phrases."

# Compress it (JSON output shows per-stage savings)
tokensqueezer compress --json "In order to save tokens, remove very long phrases."

# Fit it to 20 tokens
tokensqueezer compress --budget 20 "Your long prompt here..."

# Run the benchmark and write outputs/report.md
tokensqueezer benchmark

# Run the full demo (worked example + budget plans + benchmark)
python examples/demo.py
```

## Python API

```python
from tokensqueezer import compress, count_tokens, plan, PromptPart, run_benchmark

count_tokens("Hello, world!")  # 9

result = compress("In order to save tokens, remove very long phrases.", budget=20)
result.compressed              # 'To save tokens, remove long phrases.'
result.pct_saved               # 25.0

parts = [
    PromptPart("system", "You are a helpful assistant.", priority=1),
    PromptPart("context", "Long retrieved context...", priority=3),
    PromptPart("question", "What drove revenue growth?", priority=1),
]
planned = plan(parts, budget=120)
planned.tokens_used            # <= 120, guaranteed
planned.parts[1].action        # 'compress', 'drop', 'truncate' or 'keep'

bench = run_benchmark()
bench.mean_pct_saved           # 18.7
bench.mean_keyword_retention   # 1.0
```

## How the compression works

1. **Normalize** — collapse whitespace, fix spacing around punctuation.
2. **Fillers** — drop filler words (`very`, `really`, `just`, `actually`...) and replace wordy phrases (`in order to` → `to`, `due to the fact that` → `because`, `a large number of` → `many`). Filler words are removed before phrase matching, so interrupted phrases (`it is really important to note that` → `Note:`) still match. Articles are repaired afterwards (`a really amazing` → `an amazing`).
3. **Shorthand** — replace repeated long terms with standard abbreviations (`artificial intelligence` → `AI`, `machine learning` → `ML`, `large language model` → `LLM`). Ambiguous rewrites are avoided: `that is` as a relative pronoun is never rewritten.
4. **Dedup** — drop sentences that restate an earlier one (content-word Jaccard ≥ 0.5, or overlap coefficient ≥ 0.8). The first sentence is never dropped — it usually carries the instruction.
5. **Truncate** — keep the highest-value sentences (content-word density + first-sentence bonus) that fit the budget; hard-truncate a single over-long sentence at a token boundary.

## Project structure

```
src/tokensqueezer/
  tokenizer.py   byte-level BPE (train + encode/decode)
  merges.json    1000 merges trained on corpus.txt
  corpus.txt     bundled English training corpus
  counter.py     token counting + cost estimation
  compress.py    5-stage compression pipeline
  budget.py      multi-part budget planner
  data.py        6 benchmark samples with keywords + key facts
  benchmark.py   retention metrics + report writer
  cli.py         command-line interface
examples/demo.py deterministic demo -> outputs/report.md + outputs/summary.json
scripts/train_merges.py  regenerate merges.json from corpus.txt
tests/           59 pytest tests
```

## Verified

- **59/59 tests pass** (`pytest`), including edge cases: empty input, unicode/emoji round trips, zero budgets, single-sentence truncation, and the guarantee that a planned prompt never exceeds its budget.
- **Demo verified twice, second time in a fresh venv** created from `requirements.txt` only — byte-identical `report.md` and `summary.json`.
- No network, no API key, no secrets, no placeholders.

## License

MIT — see [LICENSE](LICENSE).
