
"""Retrieval and answer evaluation utilities for RAG systems.

Answer-level lexical scores are approximate overlap metrics, not semantic
judgments or proof that an answer is factually correct.
"""

from collections import Counter
from collections.abc import Mapping, Sequence
from math import isfinite, log2
from numbers import Real
import re


def _validate_ids(values: Sequence[str], name: str) -> None:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{name} must be a sequence of strings.")
    if any(not isinstance(value, str) for value in values):
        raise TypeError(f"all values in {name} must be strings.")


def _validate_k(k: int) -> None:
    if isinstance(k, bool) or not isinstance(k, int):
        raise TypeError("k must be an integer.")
    if k <= 0:
        raise ValueError("k must be greater than zero.")


def _validate_relevant_ids(relevant_ids: Sequence[str]) -> set[str]:
    _validate_ids(relevant_ids, "relevant_ids")
    relevant = set(relevant_ids)
    if not relevant:
        raise ValueError("relevant_ids cannot be empty.")
    return relevant


def _validate_relevance_scores(
    relevance_scores: Mapping[str, Real],
) -> dict[str, float]:
    if not isinstance(relevance_scores, Mapping):
        raise TypeError("relevance_scores must be a mapping.")

    scores: dict[str, float] = {}
    for document_id, score in relevance_scores.items():
        if not isinstance(document_id, str):
            raise TypeError("relevance score keys must be strings.")
        if isinstance(score, bool) or not isinstance(score, Real):
            raise TypeError("relevance scores must be numeric.")
        if not isfinite(float(score)) or score < 0:
            raise ValueError(
                "relevance scores must be finite and non-negative."
            )
        scores[document_id] = float(score)
    return scores


def _validate_query_collections(
    retrieved_results: Sequence[Sequence[str]],
    relevant_results: Sequence[Sequence[str]],
) -> None:
    for values, name in (
        (retrieved_results, "retrieved_results"),
        (relevant_results, "relevant_results"),
    ):
        if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
            raise TypeError(f"{name} must be a sequence.")

    if len(retrieved_results) != len(relevant_results):
        raise ValueError(
            "retrieved_results and relevant_results must have equal lengths."
        )
    if not retrieved_results:
        raise ValueError("at least one query result is required.")


def _tokens(text: str) -> list[str]:
    """Normalize text into lowercase word tokens."""
    if not isinstance(text, str):
        raise TypeError("text must be a string.")
    return re.findall(r"\w+", text.casefold(), flags=re.UNICODE)


def _validate_text(text: str, name: str) -> str:
    if not isinstance(text, str):
        raise TypeError(f"{name} must be a string.")
    if not text.strip():
        raise ValueError(f"{name} cannot be empty.")
    return text.strip()


def _normalize_contexts(contexts: str | Sequence[str]) -> list[str]:
    if isinstance(contexts, str):
        normalized = [contexts] if contexts.strip() else []
    elif isinstance(contexts, Sequence):
        if any(not isinstance(context, str) for context in contexts):
            raise TypeError("each context must be a string.")
        normalized = [context.strip() for context in contexts if context.strip()]
    else:
        raise TypeError("contexts must be a string or a sequence of strings.")

    return normalized


def _token_f1(predicted: str, reference: str) -> float:
    predicted_tokens = _tokens(predicted)
    reference_tokens = _tokens(reference)

    if not predicted_tokens or not reference_tokens:
        return 0.0

    overlap = sum(
        (Counter(predicted_tokens) & Counter(reference_tokens)).values()
    )
    precision = overlap / len(predicted_tokens)
    recall = overlap / len(reference_tokens)

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# -------------------- Retrieval metrics --------------------


def precision_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: Sequence[str],
    k: int,
) -> float:
    """Fraction of top-k result positions that are relevant."""
    _validate_ids(retrieved_ids, "retrieved_ids")
    _validate_ids(relevant_ids, "relevant_ids")
    _validate_k(k)

    top_k = list(retrieved_ids[:k])
    if not top_k:
        return 0.0

    relevant = set(relevant_ids)
    return sum(doc_id in relevant for doc_id in top_k) / k


def recall_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: Sequence[str],
    k: int,
) -> float:
    """Fraction of all relevant documents found in the top-k results."""
    _validate_ids(retrieved_ids, "retrieved_ids")
    _validate_k(k)
    relevant = _validate_relevant_ids(relevant_ids)
    return len(set(retrieved_ids[:k]) & relevant) / len(relevant)


def f1_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: Sequence[str],
    k: int,
) -> float:
    """Harmonic mean of Precision@k and Recall@k."""
    precision = precision_at_k(retrieved_ids, relevant_ids, k)
    recall = recall_at_k(retrieved_ids, relevant_ids, k)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def reciprocal_rank(
    retrieved_ids: Sequence[str],
    relevant_ids: Sequence[str],
) -> float:
    """Reciprocal rank of the first relevant result."""
    _validate_ids(retrieved_ids, "retrieved_ids")
    relevant = _validate_relevant_ids(relevant_ids)

    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def mean_reciprocal_rank(
    retrieved_results: Sequence[Sequence[str]],
    relevant_results: Sequence[Sequence[str]],
) -> float:
    """Average reciprocal rank across multiple queries."""
    _validate_query_collections(retrieved_results, relevant_results)
    scores = [
        reciprocal_rank(retrieved, relevant)
        for retrieved, relevant in zip(retrieved_results, relevant_results)
    ]
    return sum(scores) / len(scores)


def hit_rate_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: Sequence[str],
    k: int,
) -> float:
    """Return 1 if any relevant document occurs in the top-k, else 0."""
    _validate_ids(retrieved_ids, "retrieved_ids")
    _validate_k(k)
    relevant = _validate_relevant_ids(relevant_ids)
    return float(any(doc_id in relevant for doc_id in retrieved_ids[:k]))


def ndcg_at_k(
    retrieved_ids: Sequence[str],
    relevance_scores: Mapping[str, Real],
    k: int,
) -> float:
    """Normalized discounted cumulative gain for graded relevance."""
    _validate_ids(retrieved_ids, "retrieved_ids")
    _validate_k(k)
    scores = _validate_relevance_scores(relevance_scores)

    def dcg(ranked_ids: Sequence[str]) -> float:
        return sum(
            (2 ** scores.get(doc_id, 0.0) - 1) / log2(rank + 1)
            for rank, doc_id in enumerate(ranked_ids[:k], start=1)
        )

    actual = dcg(retrieved_ids)
    ideal_scores = sorted(scores.values(), reverse=True)[:k]
    ideal = sum(
        (2 ** score - 1) / log2(rank + 1)
        for rank, score in enumerate(ideal_scores, start=1)
    )
    return actual / ideal if ideal else 0.0


def average_precision(
    retrieved_ids: Sequence[str],
    relevant_ids: Sequence[str],
) -> float:
    """Average precision for one ranked query result."""
    _validate_ids(retrieved_ids, "retrieved_ids")
    relevant = _validate_relevant_ids(relevant_ids)

    hits = 0
    precision_sum = 0.0
    seen: set[str] = set()

    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant and doc_id not in seen:
            hits += 1
            precision_sum += hits / rank
            seen.add(doc_id)

    return precision_sum / len(relevant)


def mean_average_precision(
    retrieved_results: Sequence[Sequence[str]],
    relevant_results: Sequence[Sequence[str]],
) -> float:
    """Mean average precision across multiple queries."""
    _validate_query_collections(retrieved_results, relevant_results)
    scores = [
        average_precision(retrieved, relevant)
        for retrieved, relevant in zip(retrieved_results, relevant_results)
    ]
    return sum(scores) / len(scores)


def context_precision_at_k(
    retrieved_ids: Sequence[str],
    relevant_context_ids: Sequence[str],
    k: int,
) -> float:
    """Retrieval-based context precision; equivalent to Precision@k."""
    return precision_at_k(retrieved_ids, relevant_context_ids, k)


def context_recall_at_k(
    retrieved_ids: Sequence[str],
    relevant_context_ids: Sequence[str],
    k: int,
) -> float:
    """Retrieval-based context recall; equivalent to Recall@k."""
    return recall_at_k(retrieved_ids, relevant_context_ids, k)


def evaluate_retrieval(
    retrieved_results: Sequence[Sequence[str]],
    relevant_results: Sequence[Sequence[str]],
    k: int = 5,
) -> dict:
    """Return aggregate and per-query retrieval evaluation metrics."""
    _validate_query_collections(retrieved_results, relevant_results)
    _validate_k(k)

    per_query = []
    for retrieved, relevant in zip(retrieved_results, relevant_results):
        precision = precision_at_k(retrieved, relevant, k)
        recall = recall_at_k(retrieved, relevant, k)
        per_query.append(
            {
                "precision_at_k": precision,
                "recall_at_k": recall,
                "f1_at_k": f1_at_k(retrieved, relevant, k),
                "hit_rate_at_k": hit_rate_at_k(retrieved, relevant, k),
                "reciprocal_rank": reciprocal_rank(retrieved, relevant),
                "average_precision": average_precision(retrieved, relevant),
                "ndcg_at_k": ndcg_at_k(
                    retrieved,
                    {doc_id: 1.0 for doc_id in set(relevant)},
                    k,
                ),
            }
        )

    metric_names = list(per_query[0])
    summary = {
        name: sum(row[name] for row in per_query) / len(per_query)
        for name in metric_names
    }
    summary["mean_reciprocal_rank"] = summary["reciprocal_rank"]
    summary["mean_average_precision"] = summary["average_precision"]

    return {
        "query_count": len(per_query),
        "k": k,
        "metrics": summary,
        "per_query": per_query,
    }


# -------------------- Answer and context metrics --------------------


def faithfulness_score(answer: str, contexts: str | Sequence[str]) -> float:
    """Approximate answer-token support from retrieved context.

    This is a lexical proxy only. It cannot detect contradictions or verify
    factual entailment.
    """
    answer = _validate_text(answer, "answer")
    context_list = _normalize_contexts(contexts)
    answer_tokens = _tokens(answer)
    context_tokens = set(_tokens(" ".join(context_list)))

    if not answer_tokens or not context_tokens:
        return 0.0
    return sum(token in context_tokens for token in answer_tokens) / len(
        answer_tokens
    )


def answer_relevance_score(query: str, answer: str) -> float:
    """Approximate query-term coverage in the answer; not semantic relevance."""
    query = _validate_text(query, "query")
    answer = _validate_text(answer, "answer")
    query_tokens = set(_tokens(query))
    answer_tokens = set(_tokens(answer))

    if not query_tokens:
        return 0.0
    return len(query_tokens & answer_tokens) / len(query_tokens)


def context_recall_score(
    contexts: str | Sequence[str],
    reference_answer: str,
) -> float:
    """Approximate reference-answer token coverage by retrieved context."""
    reference_answer = _validate_text(reference_answer, "reference_answer")
    context_list = _normalize_contexts(contexts)
    reference_tokens = set(_tokens(reference_answer))
    context_tokens = set(_tokens(" ".join(context_list)))

    if not reference_tokens:
        return 0.0
    return len(reference_tokens & context_tokens) / len(reference_tokens)


def reference_answer_f1(answer: str, reference_answer: str) -> float:
    """Token-overlap F1 between a generated and reference answer."""
    answer = _validate_text(answer, "answer")
    reference_answer = _validate_text(reference_answer, "reference_answer")
    return _token_f1(answer, reference_answer)


def evaluate_answer(
    query: str,
    answer: str,
    contexts: str | Sequence[str],
    reference_answer: str | None = None,
) -> dict:
    """Evaluate one answer and optionally compare it with a reference answer."""
    query = _validate_text(query, "query")
    answer = _validate_text(answer, "answer")
    context_list = _normalize_contexts(contexts)

    report = {
        "faithfulness": faithfulness_score(answer, context_list),
        "answer_relevance": answer_relevance_score(query, answer),
        "context_count": len(context_list),
        "reference_answer_f1": None,
        "context_recall": None,
    }

    if reference_answer is not None:
        reference_answer = _validate_text(reference_answer, "reference_answer")
        report["reference_answer_f1"] = reference_answer_f1(
            answer, reference_answer
        )
        report["context_recall"] = context_recall_score(
            context_list, reference_answer
        )

    return report


# -------------------- Dataset evaluation and reporting --------------------


def _validate_dataset(dataset: Sequence[Mapping]) -> None:
    if isinstance(dataset, (str, bytes)) or not isinstance(dataset, Sequence):
        raise TypeError("dataset must be a sequence of mappings.")
    if not dataset:
        raise ValueError("dataset cannot be empty.")

    required = {"query", "answer", "contexts"}
    for index, example in enumerate(dataset):
        if not isinstance(example, Mapping):
            raise TypeError(f"dataset item {index} must be a mapping.")
        missing = required - set(example)
        if missing:
            raise ValueError(
                f"dataset item {index} is missing required fields: "
                f"{', '.join(sorted(missing))}."
            )


def evaluate_rag_dataset(
    dataset: Sequence[Mapping],
    k: int = 5,
) -> dict:
    """Evaluate answer quality and optional retrieval labels for a dataset.

    Each item requires:
      query: str
      answer: str
      contexts: str or sequence[str]

    Optional fields:
      reference_answer: str
      retrieved_ids: sequence[str]
      relevant_ids: sequence[str]
    """
    _validate_dataset(dataset)
    _validate_k(k)

    per_example = []
    for index, example in enumerate(dataset):
        answer_report = evaluate_answer(
            query=example["query"],
            answer=example["answer"],
            contexts=example["contexts"],
            reference_answer=example.get("reference_answer"),
        )
        row = {"example_index": index, **answer_report}

        has_retrieved = "retrieved_ids" in example
        has_relevant = "relevant_ids" in example
        if has_retrieved != has_relevant:
            raise ValueError(
                "each example must provide both retrieved_ids and relevant_ids."
            )

        if has_retrieved:
            retrieved = example["retrieved_ids"]
            relevant = example["relevant_ids"]
            row.update(
                {
                    "precision_at_k": precision_at_k(retrieved, relevant, k),
                    "recall_at_k": recall_at_k(retrieved, relevant, k),
                    "f1_at_k": f1_at_k(retrieved, relevant, k),
                    "hit_rate_at_k": hit_rate_at_k(retrieved, relevant, k),
                    "reciprocal_rank": reciprocal_rank(retrieved, relevant),
                }
            )

        per_example.append(row)

    metric_names = {
        key
        for row in per_example
        for key, value in row.items()
        if key != "example_index" and isinstance(value, Real)
        and not isinstance(value, bool)
    }
    summary = {}
    for name in sorted(metric_names):
        values = [
            float(row[name])
            for row in per_example
            if row.get(name) is not None
        ]
        if values:
            summary[name] = sum(values) / len(values)

    return {
        "example_count": len(per_example),
        "k": k,
        "metrics": summary,
        "per_example": per_example,
        "metric_notes": {
            "faithfulness": (
                "Lexical context overlap proxy; not proof of factual grounding."
            ),
            "answer_relevance": (
                "Query-term coverage proxy; not a semantic relevance judgment."
            ),
            "reference_answer_f1": (
                "Lexical token-overlap F1; paraphrases may score poorly."
            ),
            "context_recall": (
                "Reference-token coverage proxy, not claim-level recall."
            ),
        },
    }
