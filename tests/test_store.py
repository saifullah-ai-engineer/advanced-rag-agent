"""Tests for the SQLite vector store and the BM25 index."""

from advanced_rag_agent.chunking import Chunk


def test_add_and_count(empty_store, embedder):
    chunks = [
        Chunk(text="cats are lovely animals", source="a.md"),
        Chunk(text="dogs are loyal animals", source="b.md"),
    ]
    vectors = embedder.embed([c.text for c in chunks])
    ids = empty_store.add(chunks, vectors)
    assert len(ids) == 2
    assert empty_store.count() == 2


def test_dense_search_returns_closest(small_store, embedder):
    query = embedder.embed(["how do I install the package with pip"])[0]
    results = small_store.search_dense(query, top_k=2)
    assert results
    assert "install" in results[0][0].text.lower()


def test_bm25_search_returns_keyword_match(small_store):
    results = small_store.search_bm25("reciprocal rank fusion", top_k=2)
    assert results
    assert "fusion" in results[0][0].text.lower()


def test_kind_filter(small_store, embedder):
    query = embedder.embed(["add two numbers"])[0]
    results = small_store.search_dense(query, top_k=5, kind="code")
    assert results
    assert all(chunk.kind == "code" for chunk, _ in results)
    docs = small_store.search_bm25("install", top_k=5, kind="docs")
    assert all(chunk.kind == "docs" for chunk, _ in docs)


def test_persistence_reopen(tmp_path, embedder):
    from advanced_rag_agent.store import SQLiteVectorStore

    path = str(tmp_path / "persist.db")
    store = SQLiteVectorStore(path)
    chunks = [Chunk(text="hello world", source="a.md")]
    store.add(chunks, embedder.embed(["hello world"]))
    store.close()
    reopened = SQLiteVectorStore(path)
    try:
        assert reopened.count() == 1
        assert reopened.all_chunks()[0].text == "hello world"
    finally:
        reopened.close()


def test_empty_store_searches_return_nothing(empty_store, embedder):
    query = embedder.embed(["anything"])[0]
    assert empty_store.search_dense(query) == []
    assert empty_store.search_bm25("anything") == []
