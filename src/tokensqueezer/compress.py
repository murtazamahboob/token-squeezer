"""Prompt compression pipeline: do the same job with fewer tokens.

Stages, applied in order (each one is a pure function, so savings are
attributed per stage and the pipeline is fully inspectable):

1. ``normalize`` — collapse whitespace, fix spacing around punctuation.
2. ``fillers`` — drop filler words (``very``, ``really``, ``just`` ...) and
   replace wordy phrases (``in order to`` -> ``to``).
3. ``shorthand`` — replace repeated long terms with standard abbreviations
   (``artificial intelligence`` -> ``AI``).
4. ``dedup`` — drop near-duplicate sentences (content-word Jaccard >= 0.5, or
   an overlap coefficient >= 0.8 for restatements), keeping the first
   occurrence and never the first sentence.
5. ``truncate`` — only when a token budget is given: keep the highest-value
   sentences (content-word density + position bonus) that fit the budget,
   hard-truncating the best sentence if even one sentence is too long.

Nothing here calls a model or the network: every stage is a deterministic
text rewrite, and every stage reports its own token count.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .counter import default_tokenizer
from .tokenizer import BPETokenizer

# ---------------------------------------------------------------------------
# Word helpers
# ---------------------------------------------------------------------------

STOPWORDS = frozenset(
    """
    a an the and or but if then else when while of at by for with about against
    between into through during before after above below to from up down in out
    on off over under again further once here there all any both each few more
    most other some such no nor not only own same so than too very can will
    just should now is am are was were be been being have has had having do
    does did doing would could ought i you he she it we they me him her us
    them my your his its our their this that these those as of
    """.split()
)

WORD_RE = re.compile(r"[A-Za-z0-9]+")


def word_tokens(text: str) -> list[str]:
    return WORD_RE.findall(text)


def content_words(text: str) -> list[str]:
    return [w.lower() for w in word_tokens(text) if w.lower() not in STOPWORDS]


# ---------------------------------------------------------------------------
# Stage 1: normalization
# ---------------------------------------------------------------------------


def normalize_text(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)
    text = re.sub(r"([(])\s+", r"\1", text)
    text = re.sub(r"\s+([)])", r"\1", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Stage 2 & 3: phrase tables
# ---------------------------------------------------------------------------

# Wordy phrases -> tight replacements (applied before single-word fillers).
PHRASE_REPLACEMENTS: list[tuple[str, str]] = [
    ("in order to", "to"),
    ("due to the fact that", "because"),
    ("in spite of the fact that", "although"),
    ("a large number of", "many"),
    ("a small number of", "few"),
    ("at this point in time", "now"),
    ("in the event that", "if"),
    ("for the purpose of", "for"),
    ("with regard to", "about"),
    ("in the near future", "soon"),
    ("on a daily basis", "daily"),
    ("the majority of", "most"),
    ("it is important to note that", "note:"),
    ("please note that", "note:"),
    ("a great deal of", "much"),
    ("a lot of", "many"),
    ("in terms of", "for"),
    ("in the middle of", "during"),
    ("for all intents and purposes", "effectively"),
    ("in my opinion", ""),
    ("as a matter of fact", "in fact"),
]

FILLER_WORDS = frozenset(
    """
    very really just actually basically quite simply extremely totally
    literally definitely certainly obviously clearly frankly honestly
    essentially utterly absolutely truly
    """.split()
)

# Long terms -> standard abbreviations (applied after fillers).
SHORTHAND: list[tuple[str, str]] = [
    ("artificial intelligence", "AI"),
    ("machine learning", "ML"),
    ("deep learning", "DL"),
    ("natural language processing", "NLP"),
    ("large language models", "LLMs"),
    ("large language model", "LLM"),
    ("application programming interfaces", "APIs"),
    ("application programming interface", "API"),
    ("user interface", "UI"),
    ("user experience", "UX"),
    ("software development kit", "SDK"),
    ("command line interface", "CLI"),
    ("frequently asked questions", "FAQ"),
    ("for example", "e.g."),
    ("in other words", "i.e."),
    ("as soon as possible", "ASAP"),
    ("with respect to", "re"),
    ("point of view", "POV"),
]


def _replace_phrases(text: str, table: list[tuple[str, str]]) -> tuple[str, int]:
    """Case-insensitive whole-phrase replacement; returns (text, count)."""
    count = 0
    for phrase, repl in table:
        pattern = re.compile(r"\b" + re.escape(phrase) + r"\b", re.IGNORECASE)

        def _sub(match: re.Match) -> str:
            nonlocal count
            count += 1
            original = match.group(0)
            if not repl:
                return ""
            if repl.isupper() and len(repl) > 1:  # acronym: keep as-is
                return repl
            if original[0].isupper():
                return repl[0].upper() + repl[1:]
            return repl

        text = pattern.sub(_sub, text)
    return text, count


def _fix_articles(text: str) -> str:
    """Fix "a" -> "an" before vowel-initial words.

    Filler removal can create these ("a really amazing" -> "a amazing"), so
    repair them. "u" is excluded: most "u" words take "a" ("a user", "a useful
    tool"); a short list of vowel-sound "u" words is handled explicitly.
    """
    _U_EXCEPTIONS = "urgent|ugly|umbrella|under|unusual|unhappy|unable|upset|ultimate"

    def _sub(match: re.Match) -> str:
        article, word = match.group(1), match.group(2)
        fixed = "An" if article == "A" else "an"
        return f"{fixed} {word}"

    text = re.sub(r"\b([Aa]) ([aeioAEIO])", _sub, text)
    text = re.sub(rf"\b([Aa]) ({_U_EXCEPTIONS})\b", _sub, text, flags=re.IGNORECASE)
    text = re.sub(r"\b([Aa]) (hour)", _sub, text)
    return text


def remove_fillers(text: str) -> tuple[str, int]:
    """Drop filler words and wordy phrases. Returns (text, replacements).

    Filler words go first so that phrases interrupted by a filler
    ("it is really important to note that") become matchable afterwards.
    """
    pattern = re.compile(
        r"\b(" + "|".join(sorted(FILLER_WORDS)) + r")\b", re.IGNORECASE
    )
    n_words = 0

    def _drop(match: re.Match) -> str:
        nonlocal n_words
        n_words += 1
        return ""

    text = pattern.sub(_drop, text)
    # Normalize before phrase matching: dropping a word leaves double spaces
    # ("it is really important" -> "it is  important"), which would break
    # the single-space phrase patterns below.
    text = normalize_text(text)
    text, n_phrases = _replace_phrases(text, PHRASE_REPLACEMENTS)
    return _fix_articles(normalize_text(text)), n_phrases + n_words


def apply_shorthand(text: str) -> tuple[str, int]:
    """Replace long terms with standard abbreviations. Returns (text, count)."""
    text, count = _replace_phrases(text, SHORTHAND)
    return _fix_articles(normalize_text(text)), count


# ---------------------------------------------------------------------------
# Stage 4: near-duplicate sentence removal
# ---------------------------------------------------------------------------

_ABBREV = {"e.g.": "e␣g␣", "i.e.": "i␣e␣", "etc.": "etc␣", "Mr.": "Mr␣", "Dr.": "Dr␣", "vs.": "vs␣"}
_ABBREV_REV = {v: k for k, v in _ABBREV.items()}
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> list[str]:
    """Split *text* into sentences, protecting common abbreviations."""
    protected = text
    for abbr, placeholder in _ABBREV.items():
        protected = protected.replace(abbr, placeholder)
    parts = [p for p in _SENT_SPLIT.split(protected.strip()) if p]
    return [_restore(p) for p in parts]


def _restore(text: str) -> str:
    for placeholder, abbr in _ABBREV_REV.items():
        text = text.replace(placeholder, abbr)
    return text


def deduplicate_sentences(text: str, threshold: float = 0.5) -> tuple[str, int]:
    """Drop sentences that near-duplicate an earlier one. Returns (text, dropped).

    A sentence is dropped when it is a near-duplicate of an earlier kept
    sentence: Jaccard similarity of content words >= 0.5, or an overlap
    coefficient >= 0.8 (the shorter sentence's content words are almost all
    contained in the earlier one — the restatement pattern). The first
    sentence is never dropped: it usually carries the instruction.
    """
    sentences = split_sentences(text)
    if len(sentences) <= 1:
        return text, 0
    kept: list[str] = []
    kept_sets: list[set[str]] = []
    dropped = 0
    for index, sent in enumerate(sentences):
        words = set(content_words(sent)) or set(word_tokens(sent))
        duplicate = False
        if index > 0:
            for prev in kept_sets:
                inter = words & prev
                union = words | prev
                jaccard = len(inter) / len(union) if union else 0.0
                overlap = len(inter) / min(len(words), len(prev)) if words and prev else 0.0
                if jaccard >= threshold or (overlap >= 0.8 and min(len(words), len(prev)) >= 3):
                    duplicate = True
                    break
        if duplicate:
            dropped += 1
        else:
            kept.append(sent)
            kept_sets.append(words)
    return " ".join(kept), dropped


# ---------------------------------------------------------------------------
# Stage 5: budget truncation
# ---------------------------------------------------------------------------


def _sentence_value(sentence: str, index: int, total: int) -> float:
    words = word_tokens(sentence)
    if not words:
        return 0.0
    value = len(content_words(sentence)) / len(words)
    if index == 0:
        value += 0.3  # first sentence usually carries the instruction
    if index == total - 1 and total > 1:
        value += 0.05
    return value


def truncate_to_budget(
    text: str, budget: int, tokenizer: BPETokenizer
) -> tuple[str, int, bool]:
    """Keep the highest-value sentences fitting *budget* tokens.

    Returns (text, dropped_sentence_count, hard_truncated). Guarantees the
    result uses at most *budget* tokens (0 tokens when budget <= 0).
    """
    sentences = split_sentences(text)
    if budget <= 0 or not sentences:
        return "", len(sentences), False
    if tokenizer.count(text) <= budget:
        return text, 0, False

    ranked = sorted(
        range(len(sentences)),
        key=lambda i: -_sentence_value(sentences[i], i, len(sentences)),
    )
    chosen: list[int] = []
    used = 0
    for i in ranked:
        cost = tokenizer.count(sentences[i]) + (1 if chosen else 0)
        if used + cost <= budget:
            chosen.append(i)
            used += cost
    if not chosen:
        # Even the best sentence is too long: hard-truncate it at the token
        # boundary (never mid-character) and stay inside the budget.
        top = ranked[0]
        ids = tokenizer.encode(sentences[top])[:budget]
        # Decode the raw bytes with errors ignored so a slice never ends
        # mid-character (a token boundary can split a multi-byte char).
        raw = b"".join(tokenizer.id_to_token[i] for i in ids)
        truncated = raw.decode("utf-8", errors="ignore").rstrip()
        if tokenizer.count(truncated) > budget:  # safety net, never expected
            truncated = tokenizer.decode(tokenizer.encode(truncated)[:budget])
        return truncated, len(sentences) - 1, True
    chosen.sort()
    return " ".join(sentences[i] for i in chosen), len(sentences) - len(chosen), False


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------


@dataclass
class StageResult:
    name: str
    tokens_before: int
    tokens_after: int
    detail: str = ""

    @property
    def saved(self) -> int:
        return self.tokens_before - self.tokens_after


@dataclass
class CompressionResult:
    original: str
    compressed: str
    tokens_before: int
    tokens_after: int
    stages: list[StageResult] = field(default_factory=list)

    @property
    def tokens_saved(self) -> int:
        return self.tokens_before - self.tokens_after

    @property
    def pct_saved(self) -> float:
        if not self.tokens_before:
            return 0.0
        return round(100.0 * self.tokens_saved / self.tokens_before, 1)


def compress(
    text: str,
    budget: int | None = None,
    tokenizer: BPETokenizer | None = None,
) -> CompressionResult:
    """Run the full pipeline on *text* (and fit *budget* when given)."""
    tok = tokenizer or default_tokenizer()
    stages: list[StageResult] = []
    current = text
    before = tok.count(current)

    current = normalize_text(current)
    stages.append(StageResult("normalize", before, tok.count(current), "whitespace + punctuation"))
    before = stages[-1].tokens_after

    current, n = remove_fillers(current)
    stages.append(StageResult("fillers", before, tok.count(current), f"{n} filler words/phrases removed"))
    before = stages[-1].tokens_after

    current, n = apply_shorthand(current)
    stages.append(StageResult("shorthand", before, tok.count(current), f"{n} abbreviations applied"))
    before = stages[-1].tokens_after

    current, n = deduplicate_sentences(current)
    stages.append(StageResult("dedup", before, tok.count(current), f"{n} near-duplicate sentences dropped"))
    before = stages[-1].tokens_after

    if budget is not None and tok.count(current) > budget:
        current, n_dropped, hard = truncate_to_budget(current, budget, tok)
        stages.append(
            StageResult(
                "truncate",
                before,
                tok.count(current),
                f"{n_dropped} sentences dropped" + ("; hard-truncated" if hard else ""),
            )
        )

    return CompressionResult(
        original=text,
        compressed=current,
        tokens_before=tok.count(text),
        tokens_after=tok.count(current),
        stages=stages,
    )
