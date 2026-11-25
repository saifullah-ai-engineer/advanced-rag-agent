"""Configuration for the advanced RAG agent.

Every setting has a plain default and can be overridden with an environment
variable or by passing a RagConfig object. Nothing here requires a paid
service. The LLM is optional: without OPENROUTER_API_KEY the agent answers
with the extractive fallback.
"""

import os
from dataclasses import dataclass, field


@dataclass
class RagConfig:
    """All tunables for ingestion, retrieval, and generation."""

    chunk_size: int = 800
    chunk_overlap: int = 150
    code_window_lines: int = 60
    code_window_overlap: int = 10

    top_k: int = 5
    dense_k: int = 10
    bm25_k: int = 10
    rerank_top_k: int = 3
    rrf_k: int = 60
    max_retries: int = 2
    grade_coverage: float = 0.5
    grade_min_shared: int = 2

    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    db_path: str = field(default_factory=lambda: os.environ.get("RAG_DB_PATH", "rag.db"))
    openrouter_api_key: str = field(default_factory=lambda: os.environ.get("OPENROUTER_API_KEY", ""))
    openrouter_model: str = field(
        default_factory=lambda: os.environ.get(
            "OPENROUTER_MODEL", "google/gemma-4-26b-a4b-it:free"
        )
    )
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    def force_hash_embedder(self) -> bool:
        """True when the user explicitly asked for the offline hash embedder."""
        return os.environ.get("RAG_EMBEDDER", "auto").strip().lower() == "hash"
