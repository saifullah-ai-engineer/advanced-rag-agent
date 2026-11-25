"""Retrieval: hybrid dense plus BM25 search with reciprocal rank fusion,
query rewriting, reranking, and a self-corrective loop.

The loop is the part that makes this retrieval "advanced" instead of a
single nearest-neighbour lookup: after each retrieval the chunks are
graded, and if nothing relevant came back the query is rewritten and the
search runs again, up to a small retry budget.
"""

from advanced_rag_agent.chunking import Chunk
from advanced_rag_agent.config import RagConfig
from advanced_rag_agent.reranker import content_tokens


def reciprocal_rank_fusion(
    ranked_lists: list[list[Chunk]], k: int = 60
) -> list[tuple[Chunk, float]]:
    """Fuse ranked lists with RRF. Items in several lists rise to the top."""
    scores: dict[int, float] = {}
    by_id: dict[int, Chunk] = {}
    for ranked in ranked_lists:
        for rank, chunk in enumerate(ranked):
            key = chunk.id if chunk.id is not None else id(chunk)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
            by_id[key] = chunk
    fused = sorted(scores.items(), key=lambda pair: -pair[1])
    return [(by_id[key], score) for key, score in fused]


def rewrite_query_variants(query: str, llm=None, count: int = 3) -> list[str]:
    """Produce 2 to 3 search variants of a question.

    With an LLM available the variants come from the model. Without one,
    the fallback is deterministic keyword expansion: the question itself,
    its content words only, and its key terms joined with the original
    question words.
    """
    query = query.strip()
    if llm is not None:
        try:
            prompt = (
                "Rewrite this question into 3 short search queries, one per line, "
                "no numbering, no extra text.\n\nQuestion: " + query
            )
            lines = [line.strip() for line in llm.generate(prompt).splitlines() if line.strip()]
            variants = []
            for line in lines:
                cleaned = line.lstrip("0123456789.-) ").strip()
                if cleaned and cleaned.lower() not in [v.lower() for v in variants]:
                    variants.append(cleaned)
            if variants:
                if query not in variants:
                    variants.insert(0, query)
                return variants[:count]
        except Exception:  # noqa: BLE001 - fall through to keyword expansion
            pass
    terms = sorted(content_tokens(query))
    variants = [query]
    if terms:
        content_only = " ".join(terms)
        if content_only.lower() != query.lower():
            variants.append(content_only)
        if len(terms) >= 2:
            expanded = f"{query} {' '.join(terms[:4])}"
            if expanded.lower() not in [v.lower() for v in variants]:
                variants.append(expanded)
    return variants[:count]


def hybrid_search(store, embedder, query: str, config: RagConfig, kind: str | None = None):
    """Dense plus BM25 search fused with RRF. Returns top_k (chunk, score)."""
    query_vector = embedder.embed([query])[0]
    dense = store.search_dense(query_vector, top_k=config.dense_k, kind=kind)
    bm25 = store.search_bm25(query, top_k=config.bm25_k, kind=kind)
    fused = reciprocal_rank_fusion(
        [[chunk for chunk, _ in dense], [chunk for chunk, _ in bm25]], k=config.rrf_k
    )
    return fused[: config.top_k]


def grade_chunk(query: str, chunk: Chunk, config: RagConfig) -> tuple[bool, float]:
    """Grade one chunk: relevant when it shares enough terms with the question.

    The chunk name and file name count, so a question naming a function
    matches the chunk that defines it.
    """
    from pathlib import Path

    query_terms = content_tokens(query)
    if not query_terms:
        return False, 0.0
    haystack = f"{chunk.name} {Path(chunk.source).name} {chunk.text}"
    chunk_terms = content_tokens(haystack)
    shared = query_terms & chunk_terms
    coverage = len(shared) / len(query_terms)
    relevant = coverage >= config.grade_coverage or len(shared) >= config.grade_min_shared
    return relevant, coverage


def retrieve_once(store, embedder, reranker, query: str, config: RagConfig, kind=None):
    """One retrieval pass: hybrid search, rerank, grade.

    Returns (reranked chunks, relevant count).
    """
    fused = hybrid_search(store, embedder, query, config, kind=kind)
    candidates = [chunk for chunk, _ in fused]
    reranked = reranker.rerank(query, candidates, top_k=config.rerank_top_k)
    relevant = sum(1 for chunk, _ in reranked if grade_chunk(query, chunk, config)[0])
    return reranked, relevant


def retrieve_with_correction(store, embedder, reranker, query: str, config: RagConfig, kind=None, llm=None):
    """Retrieve, grade, and retry with a rewritten query when nothing fits.

    Returns (chunks, used_query, attempts, relevant_count) where chunks are
    the reranked (chunk, score) pairs from the best attempt.
    """
    variants = rewrite_query_variants(query, llm=llm, count=config.max_retries + 1)
    best: list[tuple[Chunk, float]] = []
    best_relevant = -1
    used_query = query
    attempts = 0
    for attempt, variant in enumerate(variants, start=1):
        attempts = attempt
        reranked, relevant = retrieve_once(store, embedder, reranker, variant, config, kind=kind)
        if relevant > best_relevant:
            best = reranked
            best_relevant = relevant
            used_query = variant
        if relevant > 0:
            break
    return best, used_query, attempts, max(best_relevant, 0)
