"""Tests for the rerankers."""

from advanced_rag_agent.chunking import Chunk
from advanced_rag_agent.reranker import LexicalReranker, get_reranker


def test_lexical_reranker_exact_match_first():
    reranker = LexicalReranker()
    cooking = Chunk(text="How to bake bread at home with flour and yeast.", source="a.md")
    install = Chunk(text="Install the package with pip install advanced-rag-agent.", source="b.md")
    ranked = reranker.rerank("how do I install the package", [cooking, install], top_k=2)
    assert ranked[0][0].source == "b.md"
    assert ranked[0][1] > ranked[1][1]


def test_get_reranker_falls_back_to_lexical(config):
    reranker = get_reranker(config, prefer_cross=False)
    assert isinstance(reranker, LexicalReranker)
    chunks = [Chunk(text="hello world", source="a.md")]
    assert reranker.rerank("hello", chunks, top_k=1)[0][0].source == "a.md"
