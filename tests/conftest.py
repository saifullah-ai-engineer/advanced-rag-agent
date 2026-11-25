"""Shared fixtures: a hash embedder and small preloaded stores."""

from pathlib import Path

import pytest

from advanced_rag_agent.config import RagConfig
from advanced_rag_agent.embeddings import HashEmbedder
from advanced_rag_agent.loaders import load_path
from advanced_rag_agent.store import SQLiteVectorStore

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"


@pytest.fixture
def config():
    return RagConfig()


@pytest.fixture
def embedder(config):
    return HashEmbedder(dim=config.embedding_dim)


@pytest.fixture
def doc_text():
    return """# First heading

Install the package with pip install advanced-rag-agent. The package
needs Python 3.10 or newer and works offline with the hash embedder.

# Second heading

Retrieval combines dense search with BM25 keyword search. Reciprocal
rank fusion merges the two ranked lists into one.
"""


@pytest.fixture
def code_text():
    return '''"""A tiny sample module."""


def add(a, b):
    """Return the sum of a and b."""
    return a + b


def multiply(a, b):
    """Return the product of a and b."""
    return a * b


class Calculator:
    """A small calculator class."""

    def subtract(self, a, b):
        """Return the difference of a and b."""
        return a - b
'''


@pytest.fixture
def empty_store(tmp_path):
    store = SQLiteVectorStore(str(tmp_path / "test.db"))
    yield store
    store.close()


@pytest.fixture
def small_store(tmp_path, config, embedder, doc_text, code_text):
    """A store with one docs chunk set and one code chunk set."""
    from advanced_rag_agent.chunking import chunk_code, chunk_text

    store = SQLiteVectorStore(str(tmp_path / "test.db"))
    chunks = chunk_text(doc_text, "guide.md", config.chunk_size, config.chunk_overlap)
    chunks += chunk_code(code_text, "calc.py", chunk_size=config.chunk_size)
    vectors = embedder.embed([c.text for c in chunks])
    store.add(chunks, vectors)
    yield store
    store.close()


@pytest.fixture
def examples_store(tmp_path, config, embedder):
    """A store loaded from the shipped examples corpus."""
    store = SQLiteVectorStore(str(tmp_path / "test.db"))
    chunks = load_path(EXAMPLES_DIR, config)
    vectors = embedder.embed([c.text for c in chunks])
    store.add(chunks, vectors)
    yield store
    store.close()
