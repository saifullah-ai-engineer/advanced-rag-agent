"""Tests for the CLI, including the zero-config demo."""

from pathlib import Path

from advanced_rag_agent.cli import main

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"


def test_cli_ingest_and_ask(tmp_path, capsys):
    db = str(tmp_path / "cli.db")
    assert main(["--db", db, "ingest", str(EXAMPLES_DIR)]) == 0
    out = capsys.readouterr().out
    assert "ingested" in out
    assert main(["--db", db, "ask", "How do I install advanced-rag-agent?"]) == 0
    out = capsys.readouterr().out
    assert "pip" in out.lower()
    assert "citations" in out.lower()


def test_cli_ask_empty_store(tmp_path, capsys):
    db = str(tmp_path / "empty.db")
    assert main(["--db", db, "ask", "anything"]) == 1
    assert "empty" in capsys.readouterr().out.lower()


def test_cli_eval_runs(tmp_path, capsys):
    db = str(tmp_path / "eval.db")
    assert main(["--db", db, "ingest", str(EXAMPLES_DIR)]) == 0
    capsys.readouterr()
    assert main(["--db", db, "eval"]) == 0
    out = capsys.readouterr().out
    assert "means:" in out
    assert "faithfulness" in out.lower() or "faith" in out.lower()


def test_cli_demo_runs(capsys):
    assert main(["demo"]) == 0
    out = capsys.readouterr().out
    assert "Q1:" in out
    assert "Q2:" in out
    assert "Q3:" in out
    assert "citations" in out.lower()
