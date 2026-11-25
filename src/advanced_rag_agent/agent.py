"""The LangGraph agent: route, retrieve, grade, generate.

The graph is small on purpose. The router picks the lane (docs, code, or
general). Retrieve runs one hybrid search pass. Grade checks whether the
retrieved chunks are relevant, and if they are not, the graph loops back
to retrieve with a rewritten query, up to the retry budget. Generate
answers with an LLM when one is configured, otherwise with an extractive
fallback that quotes the retrieved chunks. Every answer carries citations.
"""

import re
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from advanced_rag_agent.chunking import Chunk
from advanced_rag_agent.config import RagConfig
from advanced_rag_agent.llm import build_answer_prompt, get_llm
from advanced_rag_agent.retrieval import grade_chunk, retrieve_once, rewrite_query_variants

CODE_SIGNALS = {
    "function", "functions", "method", "methods", "class", "classes",
    "def", "import", "return", "variable", "variables", "error", "errors",
    "traceback", "exception", "bug", "code", "implement", "implementation",
    "refactor", "algorithm", "parameter", "parameters", "argument", "call",
    "module", "package", "decorator", "lambda", "loop", "recursion",
}

DOCS_SIGNALS = {
    "install", "installation", "setup", "guide", "usage", "configure",
    "configuration", "config", "tutorial", "getting", "started", "quickstart",
    "readme", "document", "documentation", "explain", "what", "why", "overview",
    "feature", "features", "limit", "limits", "requirement", "requirements",
}


class AgentState(TypedDict, total=False):
    """State carried through the graph."""

    question: str
    lane: str
    variants: list[str]
    chunks: list[Chunk]
    relevant: int
    attempts: int
    answer: str
    citations: list[str]


def route_question(question: str) -> str:
    """Classify a question as docs, code, or general by keyword signals.

    Code signals count double because generic question words like "what"
    also appear in code questions.
    """
    words = set(re.findall(r"[a-z0-9_.]+", question.lower()))
    code_score = 2 * len(words & CODE_SIGNALS)
    docs_score = len(words & DOCS_SIGNALS)
    if code_score > docs_score:
        return "code"
    if docs_score > 0:
        return "docs"
    return "general"


def _best_sentences(chunk: Chunk, question: str, limit: int = 2) -> list[str]:
    """Pick the sentences in a chunk that best match the question."""
    from advanced_rag_agent.reranker import content_tokens

    question_terms = content_tokens(question)
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", chunk.text) if s.strip()]
    if not sentences:
        return [chunk.text[:200]]
    scored = []
    for sentence in sentences:
        terms = content_tokens(sentence)
        shared = len(question_terms & terms) if question_terms else 0
        scored.append((shared, sentence))
    scored.sort(key=lambda pair: -pair[0])
    return [sentence for _, sentence in scored[:limit]]


def extractive_answer(question: str, chunks: list[Chunk]) -> tuple[str, list[str]]:
    """Compose an answer from retrieved chunks, with file and line citations."""
    if not chunks:
        return (
            "I could not find relevant information in the indexed corpus.",
            [],
        )
    citations = [chunk.citation() for chunk in chunks]
    parts: list[str] = []
    for chunk in chunks[:3]:
        sentences = _best_sentences(chunk, question)
        label = chunk.name or chunk.heading or chunk.source
        parts.append(
            f"From {chunk.source} (lines {chunk.start_line}-{chunk.end_line}, {label}): "
            + " ".join(sentences)
            + f" [{chunk.citation()}]"
        )
    return "\n\n".join(parts), citations


class RagAgent:
    """Wires the store, embedder, reranker, and optional LLM into a graph."""

    def __init__(self, store, embedder, reranker, config: RagConfig, llm=None):
        self.store = store
        self.embedder = embedder
        self.reranker = reranker
        self.config = config
        self.llm = llm if llm is not None else get_llm(config)
        self.graph = self._build_graph()

    def _build_graph(self):
        config = self.config

        def router(state: AgentState) -> dict:
            lane = route_question(state["question"])
            variants = rewrite_query_variants(state["question"], llm=self.llm)
            return {"lane": lane, "variants": variants, "attempts": 0, "relevant": 0}

        def retrieve(state: AgentState) -> dict:
            attempt = state.get("attempts", 0) + 1
            variants = state.get("variants") or [state["question"]]
            query = variants[min(attempt - 1, len(variants) - 1)]
            kind = state["lane"] if state["lane"] in ("docs", "code") else None
            reranked, _ = retrieve_once(self.store, self.embedder, self.reranker, query, config, kind=kind)
            return {"chunks": [chunk for chunk, _ in reranked], "attempts": attempt}

        def grade(state: AgentState) -> dict:
            variants = state.get("variants") or [state["question"]]
            attempt = state.get("attempts", 1)
            query = variants[min(attempt - 1, len(variants) - 1)]
            relevant = sum(
                1 for chunk in state.get("chunks", []) if grade_chunk(query, chunk, config)[0]
            )
            return {"relevant": relevant}

        def generate(state: AgentState) -> dict:
            chunks = state.get("chunks", [])
            if self.llm is not None:
                prompt = build_answer_prompt(state["question"], chunks)
                try:
                    answer = self.llm.generate(prompt)
                    citations = [chunk.citation() for chunk in chunks]
                    return {"answer": answer, "citations": citations}
                except Exception:  # noqa: BLE001 - fall back to extractive
                    pass
            answer, citations = extractive_answer(state["question"], chunks)
            return {"answer": answer, "citations": citations}

        def should_retry(state: AgentState) -> str:
            if state.get("relevant", 0) == 0 and state.get("attempts", 0) <= config.max_retries:
                return "retrieve"
            return "generate"

        graph = StateGraph(AgentState)
        graph.add_node("router", router)
        graph.add_node("retrieve", retrieve)
        graph.add_node("grade", grade)
        graph.add_node("generate", generate)
        graph.add_edge(START, "router")
        graph.add_edge("router", "retrieve")
        graph.add_edge("retrieve", "grade")
        graph.add_conditional_edges("grade", should_retry, {"retrieve": "retrieve", "generate": "generate"})
        graph.add_edge("generate", END)
        return graph.compile()

    def ask(self, question: str) -> dict:
        """Run the graph and return the answer, citations, lane, and attempts."""
        result = self.graph.invoke({"question": question})
        return {
            "question": question,
            "answer": result.get("answer", ""),
            "citations": result.get("citations", []),
            "lane": result.get("lane", "general"),
            "attempts": result.get("attempts", 0),
            "relevant": result.get("relevant", 0),
        }


def build_graph(store, embedder, reranker, config: RagConfig, llm=None) -> RagAgent:
    """Build a ready-to-use agent from its parts."""
    return RagAgent(store, embedder, reranker, config, llm=llm)


def answer_question(agent: RagAgent, question: str) -> dict:
    """Ask one question through a built agent."""
    return agent.ask(question)
