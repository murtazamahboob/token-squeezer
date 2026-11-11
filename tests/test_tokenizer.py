"""Tests for the BPE tokenizer."""

import pytest

from tokensqueezer.tokenizer import BPETokenizer, train_bpe

CORPUS = "the cat sat on the mat. the dog ran to the park. cats and dogs play together."


def test_train_bpe_is_deterministic():
    assert train_bpe(CORPUS, 10) == train_bpe(CORPUS, 10)


def test_train_bpe_reduces_token_count():
    tok_plain = BPETokenizer([])
    tok_trained = BPETokenizer(train_bpe(CORPUS, 10))
    assert tok_trained.count(CORPUS) < tok_plain.count(CORPUS)


def test_default_tokenizer_loads():
    tok = BPETokenizer.default()
    assert len(tok.merges) == 1000
    assert tok.vocab_size == 1256


def test_encode_decode_roundtrip():
    tok = BPETokenizer.default()
    samples = [
        "Hello, world!",
        "Large language models read text as tokens, not as words.",
        "The quick brown fox jumps over the lazy dog.",
        "",
        "   ",
        "Café naïve — ünïcode ✓ emoji 🚀",
        "1234567890 + 42 = 43?",
        "line one\nline two\n\nline three",
        "a" * 200,
        "don't can't won't it's",
    ]
    for text in samples:
        assert tok.decode(tok.encode(text)) == text


def test_known_token_counts():
    tok = BPETokenizer.default()
    assert tok.count("Hello, world!") == 9
    assert tok.count("The quick brown fox jumps over the lazy dog.") == 22
    assert tok.count("The cat sat on the mat.") == 9
    assert tok.count("") == 0


def test_encode_is_deterministic():
    tok = BPETokenizer.default()
    text = "Determinism matters for token counting."
    assert tok.encode(text) == tok.encode(text)


def test_count_matches_encode_length():
    tok = BPETokenizer.default()
    text = "Count me in, please."
    assert tok.count(text) == len(tok.encode(text))


def test_decode_unknown_id_raises():
    tok = BPETokenizer.default()
    with pytest.raises(KeyError):
        tok.decode([tok.vocab_size + 999])
