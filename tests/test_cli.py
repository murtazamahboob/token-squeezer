"""Tests for the CLI (in-process, plus one subprocess smoke test)."""

import io
import json
import subprocess
import sys

import pytest

from tokensqueezer import cli


def test_count_command(capsys):
    assert cli.main(["count", "Hello, world!"]) == 0
    out = capsys.readouterr().out
    assert "tokens: 9" in out
    assert "words: 2" in out


def test_count_from_stdin(monkeypatch, capsys):
    monkeypatch.setattr(sys, "stdin", io.StringIO("The cat sat on the mat."))
    assert cli.main(["count", "-"]) == 0
    assert "tokens: 9" in capsys.readouterr().out


def test_compress_json(capsys):
    text = "In order to save tokens, remove very long phrases due to the fact that they are wordy."
    assert cli.main(["compress", "--json", text]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["tokens_after"] < data["tokens_before"]
    assert data["compressed"] == "To save tokens, remove long phrases because they are wordy."
    assert [s["name"] for s in data["stages"]] == ["normalize", "fillers", "shorthand", "dedup"]


def test_compress_budget(capsys):
    text = "We are building a really amazing artificial intelligence writing assistant."
    assert cli.main(["compress", "--budget", "10", text]) == 0
    out = capsys.readouterr()
    assert "tokens" in out.err


def test_benchmark_command(tmp_path, capsys):
    assert cli.main(["benchmark", "--out", str(tmp_path)]) == 0
    assert (tmp_path / "report.md").exists()
    assert (tmp_path / "summary.json").exists()
    assert "samples:" in capsys.readouterr().out


def test_budget_command(tmp_path, capsys):
    parts = [
        {"name": "system", "text": "You are helpful.", "priority": 1},
        {"name": "context", "text": "Revenue grew 12 percent this quarter.", "priority": 3},
        {"name": "question", "text": "By how much did revenue grow?", "priority": 1},
    ]
    parts_file = tmp_path / "parts.json"
    parts_file.write_text(json.dumps(parts), encoding="utf-8")
    assert cli.main(["budget", "--budget", "30", "--parts", str(parts_file)]) == 0
    out = capsys.readouterr().out
    assert "system: keep" in out
    assert "context: drop" in out
    assert "total:" in out


def test_no_subcommand_exits_nonzero():
    with pytest.raises(SystemExit) as exc:
        cli.main([])
    assert exc.value.code != 0


def test_installed_entry_point_smoke():
    # End-to-end: the console script installed by `pip install -e .` works.
    proc = subprocess.run(
        [sys.executable, "-m", "tokensqueezer.cli", "count", "Hello, world!"],
        capture_output=True,
        text=True,
        cwd=str(__file__).rsplit("/tests/", 1)[0],
    )
    assert proc.returncode == 0
    assert "tokens: 9" in proc.stdout
