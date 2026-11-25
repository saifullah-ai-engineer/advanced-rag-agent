"""Tests for the router and the LangGraph agent."""

from advanced_rag_agent.agent import RagAgent, route_question
from advanced_rag_agent.reranker import LexicalReranker


def test_router_docs():
    assert route_question("How do I install advanced-rag-agent?") == "docs"
    assert route_question("What is reciprocal rank fusion?") == "docs"


def test_router_code():
    assert route_question("What does the split_text function do?") == "code"
    assert route_question("How does the Calculator class subtract numbers?") == "code"


def test_router_general():
    assert route_question("Tell me about the project") == "general"


def _make_agent(store, embedder, config):
    return RagAgent(store, embedder, LexicalReranker(), config, llm=None)


def test_agent_docs_question_answers_with_citations(examples_store, embedder, config):
    agent = _make_agent(examples_store, embedder, config)
    result = agent.ask("How do I install advanced-rag-agent?")
    assert result["lane"] == "docs"
    assert result["answer"]
    assert result["citations"]
    assert any("getting-started.md" in c for c in result["citations"])
    assert "pip" in result["answer"].lower()


def test_agent_code_question_routes_to_code_lane(examples_store, embedder, config):
    agent = _make_agent(examples_store, embedder, config)
    result = agent.ask("What does the split_text function do in sample_module.py?")
    assert result["lane"] == "code"
    assert result["citations"]
    assert any("sample_module.py" in c for c in result["citations"])


def test_agent_empty_store_answer(empty_store, embedder, config):
    agent = _make_agent(empty_store, embedder, config)
    result = agent.ask("How do I install advanced-rag-agent?")
    assert "could not find" in result["answer"].lower()
    assert result["citations"] == []
