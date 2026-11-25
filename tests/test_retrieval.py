"""Tests for hybrid retrieval, RRF, rewriting, grading, and correction."""

from advanced_rag_agent.chunking import Chunk
from advanced_rag_agent.reranker import LexicalReranker
from advanced_rag_agent.retrieval import (
    grade_chunk,
    hybrid_search,
    reciprocal_rank_fusion,
    retrieve_with_correction,
    rewrite_query_variants,
)


def test_rrf_prefers_item_in_both_lists():
    shared = Chunk(text="shared", source="a.md", id=1)
    dense_only = Chunk(text="dense", source="b.md", id=2)
    bm25_only = Chunk(text="bm25", source="c.md", id=3)
    fused = reciprocal_rank_fusion([[shared, dense_only], [shared, bm25_only]])
    assert fused[0][0].id == 1


def test_hybrid_search_finds_relevant(examples_store, embedder, config):
    results = hybrid_search(examples_store, embedder, "How do I install advanced-rag-agent?", config)
    assert results
    assert any("getting-started.md" in chunk.source for chunk, _ in results)


def test_rewrite_query_variants_offline(config):
    variants = rewrite_query_variants("How do I install advanced-rag-agent?", llm=None)
    assert len(variants) == 3
    assert variants[0] == "How do I install advanced-rag-agent?"
    assert len(set(v.lower() for v in variants)) == 3


def test_grade_relevant_and_irrelevant(config):
    relevant = Chunk(text="Install the package with pip install advanced-rag-agent.", source="a.md")
    irrelevant = Chunk(text="The weather in Oslo is cold and rainy today.", source="b.md")
    assert grade_chunk("How do I install advanced-rag-agent?", relevant, config)[0] is True
    assert grade_chunk("How do I install advanced-rag-agent?", irrelevant, config)[0] is False


def test_retrieve_with_correction_returns_chunks_and_attempts(examples_store, embedder, config):
    reranker = LexicalReranker()
    chunks, used_query, attempts, relevant = retrieve_with_correction(
        examples_store, embedder, reranker, "What is reciprocal rank fusion?", config
    )
    assert chunks
    assert attempts >= 1
    assert relevant >= 1
    assert used_query


def test_retrieve_empty_store_returns_empty(empty_store, embedder, config):
    reranker = LexicalReranker()
    chunks, _, attempts, relevant = retrieve_with_correction(
        empty_store, embedder, reranker, "anything at all", config
    )
    assert chunks == []
    assert relevant == 0
    assert attempts == config.max_retries + 1
