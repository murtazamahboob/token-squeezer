"""Tests for the compression pipeline."""

from tokensqueezer.compress import (
    apply_shorthand,
    compress,
    deduplicate_sentences,
    normalize_text,
    remove_fillers,
    split_sentences,
    truncate_to_budget,
)
from tokensqueezer.counter import default_tokenizer

VERBOSE = (
    "We are building a really amazing artificial intelligence writing assistant that is "
    "actually designed to help users write better content in a very fast and efficient way. "
    "The artificial intelligence writing assistant uses machine learning and deep learning "
    "to understand what the user wants to say. In order to get started, the user just needs "
    "to sign up for an account. Due to the fact that many users are very busy, the "
    "artificial intelligence writing assistant saves a large number of hours every week."
)


def test_normalize_collapses_whitespace():
    assert normalize_text("a   b\n\n  c . d") == "a b c. d"


def test_normalize_empty():
    assert normalize_text("   ") == ""


def test_remove_fillers_drops_words():
    text, n = remove_fillers("This is very really just actually amazing.")
    assert "very" not in text.lower()
    assert "really" not in text.lower()
    assert "just" not in text.lower()
    assert "actually" not in text.lower()
    assert "amazing" in text
    assert n >= 4


def test_remove_fillers_replaces_phrases():
    text, _ = remove_fillers("In order to save tokens due to the fact that prompts are long.")
    assert text == "To save tokens because prompts are long."


def test_remove_fillers_preserves_meaning_words():
    text, _ = remove_fillers("The model is very fast and really accurate.")
    assert "fast" in text and "accurate" in text


def test_shorthand_replaces_terms():
    text, n = apply_shorthand(
        "Artificial intelligence and machine learning use large language models."
    )
    assert text == "AI and ML use LLMs."
    assert n == 3


def test_shorthand_keeps_sentence_case():
    text, _ = apply_shorthand("For example, natural language processing is useful.")
    assert text.startswith("E.g., NLP is useful.")


def test_shorthand_does_not_rewrite_relative_that_is():
    # "that is" as a relative pronoun is not a clarification: keep it.
    text, _ = apply_shorthand("The assistant that is designed to help users.")
    assert "that is designed" in text


def test_filler_removal_fixes_articles():
    text, _ = remove_fillers("This is a really amazing and a very important result.")
    assert "an amazing" in text
    assert "an important" in text


def test_filler_removal_keeps_a_before_consonant_sounds():
    text, _ = remove_fillers("It is a really useful tool for a user.")
    assert "a useful tool" in text
    assert "a user" in text


def test_phrase_replacement_after_filler_removal():
    # "really" interrupts the phrase in the original; it must still match.
    text, _ = remove_fillers("It is really important to note that the price rose.")
    assert text.startswith("Note: the price rose.")


def test_split_sentences_protects_abbreviations():
    sentences = split_sentences("Use AI, e.g. machine learning, for this task. It works well.")
    assert len(sentences) == 2
    assert "e.g." in sentences[0]


def test_dedup_removes_exact_duplicate():
    text = "The cat sat on the mat. The cat sat on the mat. The dog barked."
    result, dropped = deduplicate_sentences(text)
    assert dropped == 1
    assert result.count("The cat sat on the mat.") == 1


def test_dedup_removes_near_duplicate():
    text = (
        "The quarterly revenue grew by twelve percent this quarter. "
        "Quarterly revenue grew twelve percent this quarter. "
        "Costs stayed flat."
    )
    _, dropped = deduplicate_sentences(text)
    assert dropped == 1


def test_dedup_keeps_distinct_sentences():
    text = "Cats are cute. Dogs are loyal. Birds can fly."
    result, dropped = deduplicate_sentences(text)
    assert dropped == 0
    assert result == text


def test_compress_reduces_tokens():
    result = compress(VERBOSE)
    assert result.tokens_after < result.tokens_before
    assert result.pct_saved >= 15
    assert "AI" in result.compressed or "artificial intelligence" in result.compressed.lower()


def test_compress_stage_stats_sum():
    result = compress(VERBOSE)
    total_saved = sum(stage.saved for stage in result.stages)
    assert total_saved == result.tokens_saved
    assert [s.name for s in result.stages] == ["normalize", "fillers", "shorthand", "dedup"]


def test_compress_empty():
    result = compress("")
    assert result.compressed == ""
    assert result.tokens_before == 0
    assert result.tokens_after == 0
    assert result.pct_saved == 0.0


def test_compress_whitespace_only():
    result = compress("   \n\n  ")
    assert result.compressed == ""
    assert result.tokens_after == 0


def test_compress_budget_respected():
    result = compress(VERBOSE, budget=30)
    assert result.tokens_after <= 30
    assert result.compressed


def test_compress_budget_larger_than_text():
    result = compress("Short text.", budget=1000)
    assert result.tokens_after == result.tokens_before


def test_truncate_hard_case_single_long_sentence():
    tok = default_tokenizer()
    sentence = " ".join(["word"] * 100) + "."
    text, dropped, hard = truncate_to_budget(sentence, 10, tok)
    assert tok.count(text) <= 10
    assert text
    assert hard is True


def test_truncate_zero_budget():
    tok = default_tokenizer()
    text, _, _ = truncate_to_budget("One sentence. Two sentences.", 0, tok)
    assert text == ""


def test_truncate_keeps_first_sentence_when_possible():
    tok = default_tokenizer()
    text = (
        "Summarize this report in three points. "
        "The report is long and boring and full of filler words that nobody needs. "
        "It also repeats itself many times in many different ways."
    )
    result, _, _ = truncate_to_budget(text, 25, tok)
    assert result.startswith("Summarize this report")


def test_truncate_never_exceeds_budget():
    tok = default_tokenizer()
    for budget in [1, 2, 5, 10, 20, 50]:
        text, _, _ = truncate_to_budget(VERBOSE, budget, tok)
        assert tok.count(text) <= budget
