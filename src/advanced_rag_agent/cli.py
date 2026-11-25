"""Command line interface: ingest, ask, eval, demo.

Usage:
    rag-agent ingest <path> [--db rag.db]
    rag-agent ask "your question" [--db rag.db]
    rag-agent eval [--db rag.db]
    rag-agent demo

The demo needs no key and no model download: it uses the hash embedder
and a temporary database.
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path

from advanced_rag_agent.agent import build_graph
from advanced_rag_agent.config import RagConfig
from advanced_rag_agent.embeddings import get_embedder
from advanced_rag_agent.evaluation import run_eval
from advanced_rag_agent.loaders import load_path
from advanced_rag_agent.reranker import get_reranker
from advanced_rag_agent.store import SQLiteVectorStore

DEMO_QUESTIONS = [
    "How do I install advanced-rag-agent?",
    "What is reciprocal rank fusion and why does it help retrieval?",
    "What does the split_text function do in sample_module.py?",
]


def find_examples_dir() -> Path:
    """Locate the shipped examples, from the working dir or the repo root."""
    candidates = [
        Path.cwd() / "examples",
        Path(__file__).resolve().parents[2] / "examples",
    ]
    for candidate in candidates:
        if (candidate / "eval_set.json").exists():
            return candidate
    raise FileNotFoundError(
        "Could not find the examples directory. Run from the repo root "
        "or set the working directory to the checkout."
    )


def _build_agent(db_path: str, config: RagConfig, prefer_real: bool = True):
    store = SQLiteVectorStore(db_path)
    embedder = get_embedder(config, prefer_real=prefer_real)
    reranker = get_reranker(config, prefer_cross=prefer_real)
    agent = build_graph(store, embedder, reranker, config)
    return agent, store


def cmd_ingest(args, config: RagConfig) -> int:
    chunks = load_path(args.path, config)
    if not chunks:
        print(f"no supported files found under {args.path}")
        return 1
    embedder = get_embedder(config, prefer_real=True)
    store = SQLiteVectorStore(args.db)
    try:
        batch_size = 32
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            vectors = embedder.embed([c.text for c in batch])
            store.add(batch, vectors)
        print(f"ingested {len(chunks)} chunks into {args.db}")
        print(f"store now holds {store.count()} chunks")
    finally:
        store.close()
    return 0


def cmd_ask(args, config: RagConfig) -> int:
    agent, store = _build_agent(args.db, config)
    try:
        if store.count() == 0:
            print(f"the store at {args.db} is empty. Run 'rag-agent ingest <path>' first.")
            return 1
        result = agent.ask(args.question)
        print(f"lane: {result['lane']} (attempts: {result['attempts']}, relevant: {result['relevant']})")
        print()
        print(result["answer"])
        print()
        if result["citations"]:
            print("citations:")
            for citation in result["citations"]:
                print(f"  - {citation}")
    finally:
        store.close()
    return 0


def cmd_eval(args, config: RagConfig) -> int:
    examples_dir = find_examples_dir()
    eval_set = json.loads((examples_dir / "eval_set.json").read_text())
    agent, store = _build_agent(args.db, config)
    try:
        if store.count() == 0:
            print(f"the store at {args.db} is empty. Run 'rag-agent ingest <path>' first.")
            return 1
        result = run_eval(agent.ask, eval_set, store)
        print(f"{'question':<55} {'faith':>6} {'relev':>6} {'prec':>6} {'rec':>6} {'kw':>6}")
        for item in result["items"]:
            print(
                f"{item['question'][:55]:<55} "
                f"{item['faithfulness']:>6.2f} {item['answer_relevancy']:>6.2f} "
                f"{item['context_precision']:>6.2f} {item['context_recall']:>6.2f} "
                f"{item['keyword_coverage']:>6.2f}"
            )
        print()
        means = result["means"]
        print("means: " + ", ".join(f"{key}={value:.2f}" for key, value in means.items()))
    finally:
        store.close()
    return 0


def cmd_demo(args, config: RagConfig) -> int:
    examples_dir = find_examples_dir()
    print("demo: ingesting the sample corpus with the offline hash embedder")
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = str(Path(tmpdir) / "demo.db")
        chunks = load_path(examples_dir, config)
        from advanced_rag_agent.embeddings import HashEmbedder

        embedder = HashEmbedder(dim=config.embedding_dim)
        store = SQLiteVectorStore(db_path)
        try:
            vectors = embedder.embed([c.text for c in chunks])
            store.add(chunks, vectors)
            print(f"demo: ingested {len(chunks)} chunks from {examples_dir}")
            from advanced_rag_agent.reranker import LexicalReranker

            agent = build_graph(store, embedder, LexicalReranker(), config, llm=None)
            for index, question in enumerate(DEMO_QUESTIONS, start=1):
                result = agent.ask(question)
                print()
                print(f"Q{index}: {question}")
                print(f"lane: {result['lane']} (attempts: {result['attempts']}, relevant: {result['relevant']})")
                print(result["answer"])
                if result["citations"]:
                    print("citations:")
                    for citation in result["citations"]:
                        print(f"  - {citation}")
        finally:
            store.close()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag-agent",
        description="Advanced RAG agent: document QA and code search.",
    )
    parser.add_argument("--db", default="rag.db", help="path to the SQLite store")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="ingest files or a directory")
    ingest.add_argument("path", help="file or directory to ingest")
    ingest.set_defaults(func=cmd_ingest)

    ask = sub.add_parser("ask", help="ask one question")
    ask.add_argument("question", help="the question to answer")
    ask.set_defaults(func=cmd_ask)

    evaluate = sub.add_parser("eval", help="run the shipped eval set")
    evaluate.set_defaults(func=cmd_eval)

    demo = sub.add_parser("demo", help="run the zero-config demo")
    demo.set_defaults(func=cmd_demo)
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = RagConfig(db_path=args.db)
    try:
        return args.func(args, config)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 - CLI prints errors plainly
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
