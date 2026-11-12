"""Tests for the benchmark and retention metrics."""

from tokensqueezer.benchmark import (
    fact_retention,
    keyword_retention,
    run_benchmark,
    word_overlap,
    write_report,
    write_summary_json,
)


def test_metrics_in_range():
    result = run_benchmark()
    for s in result.samples:
        assert 0.0 <= s.keyword_retention <= 1.0
        assert 0.0 <= s.fact_retention <= 1.0
        assert 0.0 <= s.word_overlap <= 1.0
        assert 0.0 <= s.pct_saved <= 100.0


def test_benchmark_saves_tokens():
    result = run_benchmark()
    assert result.total_after < result.total_before
    assert result.mean_pct_saved >= 15
    assert len(result.samples) == 6


def test_benchmark_retention_is_high():
    result = run_benchmark()
    assert result.mean_keyword_retention >= 0.7
    assert result.mean_fact_retention >= 0.7
    assert result.mean_word_overlap >= 0.3


def test_keyword_retention_edge_cases():
    assert keyword_retention([], "anything") == 1.0
    assert keyword_retention(["ai", "tokens"], "AI saves tokens") == 1.0
    assert keyword_retention(["missing"], "nothing here") == 0.0


def test_fact_retention_edge_cases():
    assert fact_retention([], "anything") == 1.0
    assert fact_retention(["the cat sat"], "The cat sat on the mat.") == 1.0
    assert fact_retention(["quantum entanglement"], "The cat sat.") == 0.0


def test_word_overlap_edge_cases():
    assert word_overlap("", "") == 1.0
    assert word_overlap("cats dogs", "cats dogs") == 1.0
    assert word_overlap("cats", "dogs") == 0.0


def test_write_report_creates_file(tmp_path):
    result = run_benchmark()
    path = write_report(result, tmp_path / "report.md")
    text = path.read_text(encoding="utf-8")
    assert "# Token Squeezer — Benchmark Report" in text
    assert "Mean keyword retention" in text
    assert "Verbose product description" in text
    summary = write_summary_json(result, tmp_path / "summary.json")
    assert summary.exists()
    assert summary.read_text().strip().startswith("{")
