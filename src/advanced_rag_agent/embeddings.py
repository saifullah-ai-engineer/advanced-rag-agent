"""Embeddings behind one small protocol.

Two embedders ship with the project:

- SentenceTransformerEmbedder: the real thing, all-MiniLM-L6-v2, loaded
  lazily so importing this module never pulls in torch.
- HashEmbedder: deterministic, offline, no model download. It hashes tokens
  into a fixed size vector. It is good enough for tests, the demo, and
  offline use, and it is the automatic fallback when the real model cannot
  be loaded.
"""

import hashlib
import re
import sys
from typing import Protocol

import numpy as np


class Embedder(Protocol):
    """Anything that turns texts into one L2-normalised row per text."""

    dim: int

    def embed(self, texts: list[str]) -> np.ndarray:
        ...


_TOKEN_RE = re.compile(r"[a-z0-9_]+")


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class HashEmbedder:
    """Deterministic bag-of-hashed-words embedder. No downloads, no keys."""

    def __init__(self, dim: int = 384):
        self.dim = dim

    def embed(self, texts: list[str]) -> np.ndarray:
        rows = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row_index, text in enumerate(texts):
            for token in _tokens(text):
                bucket = int.from_bytes(hashlib.md5(token.encode()).digest()[:4], "little") % self.dim
                rows[row_index, bucket] += 1.0
            norm = np.linalg.norm(rows[row_index])
            if norm > 0:
                rows[row_index] /= norm
        return rows


class SentenceTransformerEmbedder:
    """Real dense embeddings from sentence-transformers, loaded lazily."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is not installed. Install the ml extra "
                "with: pip install 'advanced-rag-agent[ml]'"
            ) from exc
        self._model = SentenceTransformer(model_name)
        self.dim = int(self._model.get_sentence_embedding_dimension())

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = self._model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        return np.asarray(vectors, dtype=np.float32)


def get_embedder(config, prefer_real: bool = True) -> Embedder:
    """Return the real embedder when it works, else the hash fallback.

    The fallback prints a clear warning so nobody mistakes hash embeddings
    for a production retrieval setup.
    """
    if config.force_hash_embedder():
        return HashEmbedder(dim=config.embedding_dim)
    if not prefer_real:
        return HashEmbedder(dim=config.embedding_dim)
    try:
        return SentenceTransformerEmbedder(model_name=config.embedding_model)
    except Exception as exc:  # noqa: BLE001 - any load failure falls back
        print(
            f"warning: could not load embedding model "
            f"'{config.embedding_model}' ({exc}). Falling back to the "
            f"deterministic hash embedder. Retrieval still works, but "
            f"semantic quality is lower.",
            file=sys.stderr,
        )
        return HashEmbedder(dim=config.embedding_dim)
