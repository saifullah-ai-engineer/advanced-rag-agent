"""Tests for the evaluation harness."""

import json
from pathlib import Path

from advanced_rag_agent.agent import RagAgent
from advanced_rag_agent.chunking import Chunk
from advanced_rag_agent.evaluation import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
    keyword_coverage,
    run_eval,
)
from advanced_rag_agent.reranker import LexicalReranker

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"


def test_faithfulness_supported_vs_unsupported():
    chunk = Chunk(text="Install the package with pip install advanced-rag-agent.", source="a.md")
    supported = faithfulness("Install the package with pip.", [chunk])
    assert supported == 1.0
    unsupported = faithfulness("The moon is made of green cheese.", [chunk])
    assert unsupported == 0.0


def test_answer_relevancy():
    assert answer_relevancy("how to install", "install the package with pip") == 1.0
    assert answer_relevancy("how to install", "the weather is cold") == 0.0
    assert answer_relevancy("how to install the package", "install the package") == 1.0


def test_context_precision_recall():
    assert context_precision(["a.md", "b.md"], ["a.md"]) == 0.5
    assert context_recall(["a.md", "b.md"], ["a.md"]) == 1.0
    assert context_precision([], ["a.md"]) == 0.0
    assert context_recall(["a.md"], []) == 1.0


def test_keyword_coverage():
    assert keyword_coverage("Install with pip.", ["pip", "install"]) == 1.0
    assert keyword_coverage("Install with pip.", ["pip", "docker"]) == 0.5


def test_run_eval_on_tiny_set(examples_store, embedder, config):
    agent = RagAgent(examples_store, embedder, LexicalReranker(), config, llm=None)
    eval_set = json.loads((EXAMPLES_DIR / "eval_set.json").read_text())[:2]
    result = run_eval(agent.ask, eval_set, examples_store)
    assert len(result["items"]) == 2
    for key, value in result["means"].items():
        assert 0.0 <= value <= 1.0
    assert result["items"][0]["context_recall"] == 1.0
