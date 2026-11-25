"""A local vector store on SQLite, plus a BM25 index over the same chunks.

No server and no heavy vector database. Embeddings live as float32 BLOBs
in one table, and dense search is plain cosine similarity with numpy.
The BM25 index is built in memory from the stored chunk texts and rebuilt
whenever new chunks are added.
"""

import re
import sqlite3
from pathlib import Path

import numpy as np

from advanced_rag_agent.chunking import Chunk

_TOKEN_RE = re.compile(r"[a-z0-9_]+")


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class SQLiteVectorStore:
    """Chunks plus embeddings in a single SQLite file."""

    def __init__(self, path: str | Path):
        self.path = str(path)
        self._conn = sqlite3.connect(self.path)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT NOT NULL,
                kind TEXT NOT NULL,
                name TEXT NOT NULL DEFAULT '',
                heading TEXT NOT NULL DEFAULT '',
                start_line INTEGER NOT NULL DEFAULT 1,
                end_line INTEGER NOT NULL DEFAULT 1,
                text TEXT NOT NULL,
                embedding BLOB NOT NULL
            )
            """
        )
        self._conn.commit()
        self._bm25 = None
        self._bm25_chunks: list[Chunk] = []
        self._bm25_version = -1
        self._version = 0

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "SQLiteVectorStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def clear(self) -> None:
        self._conn.execute("DELETE FROM chunks")
        self._conn.commit()
        self._version += 1

    def count(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()
        return int(row[0]) if row else 0

    def add(self, chunks: list[Chunk], embeddings: np.ndarray) -> list[int]:
        """Insert chunks with their embedding rows. Returns the new ids."""
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")
        ids: list[int] = []
        for chunk, vector in zip(chunks, embeddings):
            row = self._conn.execute(
                """
                INSERT INTO chunks
                    (source, kind, name, heading, start_line, end_line, text, embedding)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    chunk.source,
                    chunk.kind,
                    chunk.name,
                    chunk.heading,
                    chunk.start_line,
                    chunk.end_line,
                    chunk.text,
                    np.asarray(vector, dtype=np.float32).tobytes(),
                ),
            )
            ids.append(int(row.lastrowid))
        self._conn.commit()
        self._version += 1
        return ids

    def all_chunks(self) -> list[Chunk]:
        """All chunks in id order, without embeddings."""
        rows = self._conn.execute(
            "SELECT id, source, kind, name, heading, start_line, end_line, text "
            "FROM chunks ORDER BY id"
        ).fetchall()
        return [
            Chunk(
                text=row[7],
                source=row[1],
                kind=row[2],
                name=row[3],
                heading=row[4],
                start_line=row[5],
                end_line=row[6],
                id=row[0],
            )
            for row in rows
        ]

    def _load_matrix(self, kind: str | None = None) -> tuple[list[Chunk], np.ndarray]:
        """Load chunks plus their embedding matrix, optionally filtered by kind."""
        if kind is None:
            rows = self._conn.execute(
                "SELECT id, source, kind, name, heading, start_line, end_line, text, embedding "
                "FROM chunks ORDER BY id"
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT id, source, kind, name, heading, start_line, end_line, text, embedding "
                "FROM chunks WHERE kind = ? ORDER BY id",
                (kind,),
            ).fetchall()
        chunks = [
            Chunk(
                text=row[7],
                source=row[1],
                kind=row[2],
                name=row[3],
                heading=row[4],
                start_line=row[5],
                end_line=row[6],
                id=row[0],
            )
            for row in rows
        ]
        if not rows:
            return chunks, np.zeros((0, 0), dtype=np.float32)
        matrix = np.stack(
            [np.frombuffer(row[8], dtype=np.float32) for row in rows]
        ).astype(np.float32)
        return chunks, matrix

    def search_dense(
        self, query_vector: np.ndarray, top_k: int = 5, kind: str | None = None
    ) -> list[tuple[Chunk, float]]:
        """Cosine similarity search. Returns (chunk, score) pairs, best first."""
        chunks, matrix = self._load_matrix(kind)
        if not chunks:
            return []
        query = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        norms = np.linalg.norm(matrix, axis=1)
        query_norm = np.linalg.norm(query)
        if query_norm == 0:
            return [(chunk, 0.0) for chunk in chunks[:top_k]]
        scores = (matrix @ query) / np.maximum(norms * query_norm, 1e-9)
        order = np.argsort(-scores)[:top_k]
        return [(chunks[i], float(scores[i])) for i in order]

    def _ensure_bm25(self, kind: str | None = None) -> None:
        """Build or rebuild the BM25 index when the store changed."""
        if self._bm25 is not None and self._bm25_version == self._version and kind is None:
            return
        from rank_bm25 import BM25Okapi

        if kind is None:
            chunks = self.all_chunks()
        else:
            chunks = [c for c in self.all_chunks() if c.kind == kind]
        corpus = [_tokens(c.text) for c in chunks]
        self._bm25 = BM25Okapi(corpus) if corpus else None
        self._bm25_chunks = chunks
        if kind is None:
            self._bm25_version = self._version

    def search_bm25(
        self, query: str, top_k: int = 5, kind: str | None = None
    ) -> list[tuple[Chunk, float]]:
        """Keyword search with BM25. Returns (chunk, score) pairs, best first."""
        self._ensure_bm25(kind)
        if self._bm25 is None or not self._bm25_chunks:
            return []
        scores = self._bm25.get_scores(_tokens(query))
        order = np.argsort(-np.asarray(scores))[:top_k]
        return [(self._bm25_chunks[i], float(scores[i])) for i in order if scores[i] > 0]
