"""A small local evaluation harness in the RAGAS spirit, with no service.

Four metrics, all computed locally from the answer and the retrieved
chunks:

- faithfulness: share of answer sentences supported by the cited chunks.
- answer relevancy: word overlap between the question and the answer.
- context precision: share of retrieved chunks that are relevant.
- context recall: share of the labelled relevant sources that were retrieved.

Relevance in the shipped eval set is labelled by source file, so precision
and recall are computed on chunk sources, not on internal chunk ids.
"""

import re

from advanced_rag_agent.reranker import content_tokens

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_RE.split(text) if s.strip()]


def faithfulness(answer: str, chunks) -> float:
    """Share of answer sentences whose words appear in the cited chunks."""
    sentences = _sentences(answer)
    if not sentences:
        return 0.0
    chunk_words: set[str] = set()
    for chunk in chunks:
        chunk_words |= content_tokens(chunk.text)
    if not chunk_words:
        return 0.0
    supported = 0
    for sentence in sentences:
        words = content_tokens(sentence)
        if not words:
            continue
        if len(words & chunk_words) / len(words) >= 0.5:
            supported += 1
    return supported / len(sentences)


def answer_relevancy(question: str, answer: str) -> float:
    """Share of the question's content words that appear in the answer.

    A lexical proxy, not a judgement: an answer that addresses the
    question repeats its key terms. Coverage is used instead of Jaccard
    because a complete answer is long, and Jaccard would punish length.
    """
    question_terms = content_tokens(question)
    answer_terms = content_tokens(answer)
    if not question_terms or not answer_terms:
        return 0.0
    return len(question_terms & answer_terms) / len(question_terms)


def context_precision(retrieved_sources: list[str], relevant_sources: list[str]) -> float:
    """Share of retrieved sources that are labelled relevant."""
    if not retrieved_sources:
        return 0.0
    relevant = {s for s in relevant_sources}
    hits = sum(1 for source in retrieved_sources if any(source.endswith(r) for r in relevant))
    return hits / len(retrieved_sources)


def context_recall(retrieved_sources: list[str], relevant_sources: list[str]) -> float:
    """Share of labelled relevant sources that were retrieved."""
    if not relevant_sources:
        return 1.0
    hits = sum(
        1
        for relevant in relevant_sources
        if any(source.endswith(relevant) for source in retrieved_sources)
    )
    return hits / len(relevant_sources)


def keyword_coverage(answer: str, keywords: list[str]) -> float:
    """Share of expected keywords that appear in the answer."""
    if not keywords:
        return 1.0
    lowered = answer.lower()
    hits = sum(1 for keyword in keywords if keyword.lower() in lowered)
    return hits / len(keywords)


def _sources_from_citations(citations: list[str]) -> list[str]:
    """Recover chunk sources from citation strings like path:12-40."""
    sources: list[str] = []
    for citation in citations:
        if ":" in citation:
            sources.append(citation.rsplit(":", 1)[0])
        else:
            sources.append(citation)
    return sources


def run_eval(ask_fn, eval_set: list[dict], store) -> dict:
    """Run an eval set through an ask function and score every item.

    ask_fn takes a question string and returns a dict with "answer" and
    "citations". The store is used to fetch chunk texts for faithfulness.
    """
    by_source = {}
    for chunk in store.all_chunks():
        by_source.setdefault(chunk.source, []).append(chunk)
    items: list[dict] = []
    for item in eval_set:
        result = ask_fn(item["question"])
        answer = result.get("answer", "")
        citations = result.get("citations", [])
        sources = _sources_from_citations(citations)
        cited_chunks = []
        for source in sources:
            cited_chunks.extend(by_source.get(source, []))
        scores = {
            "faithfulness": faithfulness(answer, cited_chunks),
            "answer_relevancy": answer_relevancy(item["question"], answer),
            "context_precision": context_precision(sources, item.get("relevant_sources", [])),
            "context_recall": context_recall(sources, item.get("relevant_sources", [])),
            "keyword_coverage": keyword_coverage(answer, item.get("expected_keywords", [])),
        }
        items.append({"question": item["question"], **scores})
    means = {}
    if items:
        for key in items[0].keys():
            if key != "question":
                means[key] = sum(item[key] for item in items) / len(items)
    return {"items": items, "means": means}
