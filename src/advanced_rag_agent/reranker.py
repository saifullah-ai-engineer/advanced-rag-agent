"""Rerankers: a cross-encoder when the ml extra is installed, otherwise a
deterministic lexical overlap reranker that always works offline.
"""

import re
import sys

from advanced_rag_agent.chunking import Chunk

_TOKEN_RE = re.compile(r"[a-z0-9_]+")

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "do", "does", "did", "doing", "have", "has", "had", "having",
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them",
    "my", "your", "his", "its", "our", "their", "this", "that", "these", "those",
    "and", "or", "but", "if", "then", "else", "when", "at", "by", "for", "with",
    "about", "into", "through", "after", "over", "between", "out", "against",
    "during", "without", "before", "under", "around", "among", "of", "to", "in",
    "on", "from", "as", "how", "what", "which", "who", "whom", "why", "can",
    "could", "should", "would", "will", "shall", "may", "might", "please",
}


def content_tokens(text: str) -> set[str]:
    """Lowercase word tokens with stopwords removed."""
    return {t for t in _TOKEN_RE.findall(text.lower()) if t not in STOPWORDS}


def _haystack(chunk: Chunk) -> str:
    """Text used for lexical scoring: name, file name, and chunk text."""
    from pathlib import Path

    return f"{chunk.name} {Path(chunk.source).name} {chunk.text}"


class LexicalReranker:
    """Rerank by query-term coverage. Fast, deterministic, offline."""

    def rerank(self, query: str, chunks: list[Chunk], top_k: int = 3) -> list[tuple[Chunk, float]]:
        query_terms = content_tokens(query)
        scored: list[tuple[Chunk, float]] = []
        for chunk in chunks:
            chunk_terms = content_tokens(_haystack(chunk))
            if not query_terms:
                score = 0.0
            else:
                shared = query_terms & chunk_terms
                score = len(shared) / len(query_terms)
                if query.lower().strip() in chunk.text.lower():
                    score += 0.25
            scored.append((chunk, score))
        scored.sort(key=lambda pair: -pair[1])
        return scored[:top_k]


class CrossEncoderReranker:
    """Rerank with a cross-encoder model. Needs the ml extra installed."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is not installed. Install the ml extra "
                "with: pip install 'advanced-rag-agent[ml]'"
            ) from exc
        self._model = CrossEncoder(model_name)

    def rerank(self, query: str, chunks: list[Chunk], top_k: int = 3) -> list[tuple[Chunk, float]]:
        pairs = [(query, chunk.text) for chunk in chunks]
        scores = self._model.predict(pairs)
        scored = list(zip(chunks, [float(s) for s in scores]))
        scored.sort(key=lambda pair: -pair[1])
        return scored[:top_k]


def get_reranker(config, prefer_cross: bool = True):
    """Return a cross-encoder reranker when it loads, else the lexical one."""
    if config.force_hash_embedder() or not prefer_cross:
        return LexicalReranker()
    try:
        return CrossEncoderReranker(model_name=config.cross_encoder_model)
    except Exception as exc:  # noqa: BLE001 - any load failure falls back
        print(
            f"note: cross-encoder reranker unavailable ({exc}). "
            f"Using the lexical reranker.",
            file=sys.stderr,
        )
        return LexicalReranker()
