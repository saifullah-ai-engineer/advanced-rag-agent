# Advanced RAG overview

## Why naive RAG falls short

Naive retrieval embeds a question, takes the nearest chunks, and hands
them to a language model. That works until it does not: a question phrased
with different words than the documents misses the right chunk, one bad
chunk poisons the answer, and the model confidently answers from nothing.

## Hybrid retrieval

This project combines two retrievers. Dense search compares embedding
vectors with cosine similarity, which catches paraphrases and meaning.
BM25 keyword search scores exact term matches, which catches names,
identifiers, and rare words. Neither is enough alone, so the two ranked
lists are fused.

## Reciprocal rank fusion

Reciprocal rank fusion, usually shortened to RRF, merges ranked lists
without needing comparable scores. Each chunk gets a fused score of the
sum of one over a constant plus its rank in each list. A chunk that ranks
high in both dense and BM25 search rises above a chunk that ranks high
in only one. RRF is simple, has no tunable weights to fit, and is robust
when the two retrievers disagree. That is why it helps retrieval: it
rewards agreement between independent signals instead of trusting one.

## Reranking

After fusion, a reranker scores the question against each candidate chunk
directly. With the ml extra installed, a cross-encoder model reads the
question and chunk together and gives a relevance score. Without it, a
lexical reranker scores query-term coverage. Reranking fixes the common
case where a chunk is near the question in embedding space but does not
actually answer it.

## Self-correction

Retrieval is graded. If no retrieved chunk shares enough terms with the
question, the agent rewrites the query and searches again, up to 2
retries. Query rewriting produces 2 to 3 variants: the original question,
its content words only, and a keyword-expanded form. With an LLM
configured, the variants come from the model instead.

## Evaluation

The eval harness scores answers on four metrics in the RAGAS spirit:
faithfulness, answer relevancy, context precision, and context recall.
Faithfulness checks that answer sentences are supported by the cited
chunks. Answer relevancy measures word overlap between the question and
the answer. Context precision and recall compare the retrieved sources
against a small labelled set shipped in examples/eval_set.json.
