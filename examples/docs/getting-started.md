# Getting started

## Installation

Install advanced-rag-agent from source with pip:

```bash
git clone https://github.com/saifullah-ai-engineer/advanced-rag-agent
cd advanced-rag-agent
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Python 3.10 or newer is required. The core install needs no paid API
and no model download: the default embedder falls back to a deterministic
hash embedder when sentence-transformers is not installed.

## Optional machine learning extras

For real dense embeddings and cross-encoder reranking, install the ml
extra:

```bash
pip install 'advanced-rag-agent[ml]'
```

This installs sentence-transformers and downloads the all-MiniLM-L6-v2
embedding model on first use. The cross-encoder model used for reranking
is cross-encoder/ms-marco-MiniLM-L-6-v2.

## Quickstart

Ingest a folder of documents and code, then ask a question:

```bash
rag-agent ingest ./docs
rag-agent ask "How do I install advanced-rag-agent?"
```

## Configuration

The most useful settings live in RagConfig. Chunk size defaults to 800
characters with 150 characters of overlap. Retrieval returns the top 5
fused results and reranks them down to 3. The self-correction loop retries
at most 2 times. Set the RAG_DB_PATH environment variable to change where
the SQLite store lives. Set OPENROUTER_API_KEY to enable LLM generation
with a free OpenRouter model; without a key the agent answers with an
extractive fallback that quotes the retrieved chunks.
