import pytest

from src.evaluation_metrics import (
    answer_relevance_score,
    average_precision,
    context_precision_at_k,
    context_recall_at_k,
    context_recall_score,
    evaluate_answer,
    evaluate_rag_dataset,
    evaluate_retrieval,
    f1_at_k,
    faithfulness_score,
    hit_rate_at_k,
    mean_average_precision,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    reference_answer_f1,
)


# Retrieval metrics


def test_precision_at_k():
    assert precision_at_k(["A", "C", "B"], ["A", "B"], 2) == 0.5


def test_precision_at_k_uses_k_as_denominator():
    assert precision_at_k(["A"], ["A"], 3) == pytest.approx(1 / 3)


def test_precision_at_k_empty_results():
    assert precision_at_k([], ["A"], 3) == 0.0


@pytest.mark.parametrize("k", [0, -1])
def test_precision_rejects_non_positive_k(k):
    with pytest.raises(ValueError, match="greater than zero"):
        precision_at_k(["A"], ["A"], k)


@pytest.mark.parametrize("k", [1.5, True, "2"])
def test_precision_rejects_invalid_k_type(k):
    with pytest.raises(TypeError, match="integer"):
        precision_at_k(["A"], ["A"], k)


def test_precision_rejects_string_collection():
    with pytest.raises(TypeError, match="sequence"):
        precision_at_k("ABC", ["A"], 2)


def test_precision_rejects_non_string_ids():
    with pytest.raises(TypeError, match="strings"):
        precision_at_k(["A", 2], ["A"], 2)


def test_recall_at_k():
    assert recall_at_k(["A", "C", "B"], ["A", "B"], 2) == 0.5


def test_recall_counts_unique_retrieved_documents():
    assert recall_at_k(["A", "A"], ["A", "B"], 2) == 0.5


def test_recall_rejects_empty_relevant_ids():
    with pytest.raises(ValueError, match="cannot be empty"):
        recall_at_k(["A"], [], 1)


def test_f1_at_k():
    assert f1_at_k(["A", "C", "B"], ["A", "B"], 2) == pytest.approx(0.5)


def test_f1_at_k_returns_zero_when_no_hits():
    assert f1_at_k(["X"], ["A"], 1) == 0.0


def test_reciprocal_rank_first_result():
    assert reciprocal_rank(["A", "B"], ["A"]) == 1.0


def test_reciprocal_rank_second_result():
    assert reciprocal_rank(["C", "B", "A"], ["B"]) == 0.5


def test_reciprocal_rank_no_match():
    assert reciprocal_rank(["C", "D"], ["A", "B"]) == 0.0


def test_reciprocal_rank_rejects_empty_relevant_ids():
    with pytest.raises(ValueError, match="cannot be empty"):
        reciprocal_rank(["A"], [])


def test_mean_reciprocal_rank():
    retrieved = [["A", "B"], ["C", "B"], ["D", "E"]]
    relevant = [["A"], ["B"], ["E"]]

    assert mean_reciprocal_rank(retrieved, relevant) == pytest.approx(2 / 3)


def test_hit_rate_at_k():
    assert hit_rate_at_k(["C", "B"], ["B"], 2) == 1.0
    assert hit_rate_at_k(["C", "D"], ["A"], 2) == 0.0


def test_ndcg_perfect_ranking():
    assert ndcg_at_k(
        ["A", "B"],
        {"A": 3, "B": 1},
        2,
    ) == pytest.approx(1.0)


def test_ndcg_prefers_better_ranking():
    scores = {"A": 3, "B": 1}

    assert ndcg_at_k(["A", "B"], scores, 2) > ndcg_at_k(
        ["B", "A"],
        scores,
        2,
    )


def test_ndcg_zero_when_all_gains_are_zero():
    assert ndcg_at_k(["X"], {"A": 0}, 1) == 0.0


def test_ndcg_rejects_negative_relevance():
    with pytest.raises(ValueError, match="non-negative"):
        ndcg_at_k(["A"], {"A": -1}, 1)


def test_ndcg_rejects_non_numeric_relevance():
    with pytest.raises(TypeError, match="numeric"):
        ndcg_at_k(["A"], {"A": "high"}, 1)


def test_average_precision():
    assert average_precision(
        ["A", "C", "B"],
        ["A", "B"],
    ) == pytest.approx(5 / 6)


def test_average_precision_does_not_count_duplicate_hits_twice():
    # A is relevant at rank 1.
    # The duplicate A at rank 2 is ignored.
    # B is relevant at rank 3.
    #
    # AP = (1/1 + 2/3) / 2 = 5/6.
    assert average_precision(
        ["A", "A", "B"],
        ["A", "B"],
    ) == pytest.approx(5 / 6)


def test_mean_average_precision():
    retrieved = [["A", "B"], ["C", "B"]]
    relevant = [["A"], ["B"]]

    assert mean_average_precision(
        retrieved,
        relevant,
    ) == pytest.approx(0.75)


def test_mean_reciprocal_rank_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="equal lengths"):
        mean_reciprocal_rank(
            [["A"]],
            [["A"], ["B"]],
        )


def test_mean_average_precision_rejects_empty_dataset():
    with pytest.raises(ValueError, match="at least one"):
        mean_average_precision([], [])


def test_context_precision_and_recall_aliases():
    assert context_precision_at_k(
        ["A", "B"],
        ["A"],
        2,
    ) == 0.5

    assert context_recall_at_k(
        ["A", "B"],
        ["A"],
        2,
    ) == 1.0


def test_evaluate_retrieval_report():
    report = evaluate_retrieval(
        retrieved_results=[
            ["A", "C"],
            ["B", "A"],
        ],
        relevant_results=[
            ["A"],
            ["A"],
        ],
        k=2,
    )

    assert report["query_count"] == 2
    assert report["k"] == 2
    assert report["metrics"]["precision_at_k"] == pytest.approx(0.5)
    assert report["metrics"]["mean_reciprocal_rank"] == pytest.approx(0.75)
    assert len(report["per_query"]) == 2


# Answer and context metrics


def test_faithfulness_score_is_one_when_answer_tokens_are_in_context():
    assert faithfulness_score(
        "Python is useful",
        ["Python is useful for data work"],
    ) == 1.0


def test_faithfulness_score_is_zero_without_context():
    assert faithfulness_score(
        "Python is useful",
        [],
    ) == 0.0


def test_faithfulness_score_is_partial_for_partial_overlap():
    score = faithfulness_score(
        "Python is useful",
        ["Python is powerful"],
    )

    assert 0.0 < score < 1.0


def test_answer_relevance_score_measures_query_token_coverage():
    # Query tokens:
    # "what", "is", "python"
    #
    # Answer contains:
    # "is", "python"
    #
    # Therefore: 2 / 3.
    assert answer_relevance_score(
        "What is Python?",
        "Python is a programming language",
    ) == pytest.approx(2 / 3)


def test_answer_relevance_score_is_zero_without_overlap():
    assert answer_relevance_score(
        "Explain Python",
        "A banana is yellow",
    ) == 0.0


def test_context_recall_score_measures_reference_token_coverage():
    assert context_recall_score(
        ["Python is a programming language"],
        "Python is a programming language",
    ) == 1.0


def test_context_recall_score_returns_zero_without_context():
    assert context_recall_score(
        [],
        "Python is useful",
    ) == 0.0


def test_reference_answer_f1_exact_match():
    assert reference_answer_f1(
        "Python is useful",
        "Python is useful",
    ) == 1.0


def test_reference_answer_f1_partial_match():
    score = reference_answer_f1(
        "Python is useful",
        "Python is powerful",
    )

    assert 0.0 < score < 1.0


def test_reference_answer_f1_zero_for_no_overlap():
    assert reference_answer_f1(
        "red apple",
        "blue ocean",
    ) == 0.0


def test_evaluate_answer_without_reference():
    report = evaluate_answer(
        query="What is Python?",
        answer="Python is a programming language",
        contexts=["Python is a programming language"],
    )

    assert report["faithfulness"] == 1.0
    assert report["answer_relevance"] == pytest.approx(2 / 3)
    assert report["context_count"] == 1
    assert report["reference_answer_f1"] is None
    assert report["context_recall"] is None


def test_evaluate_answer_with_reference():
    report = evaluate_answer(
        query="What is Python?",
        answer="Python is a programming language",
        contexts=["Python is a programming language"],
        reference_answer="Python is a programming language",
    )

    assert report["reference_answer_f1"] == 1.0
    assert report["context_recall"] == 1.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {
            "query": "",
            "answer": "A",
            "contexts": ["A"],
        },
        {
            "query": "Q",
            "answer": "",
            "contexts": ["A"],
        },
        {
            "query": "Q",
            "answer": "A",
            "contexts": 123,
        },
    ],
)
def test_evaluate_answer_rejects_invalid_inputs(kwargs):
    with pytest.raises((TypeError, ValueError)):
        evaluate_answer(**kwargs)


def test_faithfulness_rejects_non_string_answer():
    with pytest.raises(TypeError, match="answer must be a string"):
        faithfulness_score(
            123,
            ["context"],
        )


def test_context_recall_rejects_empty_reference():
    with pytest.raises(ValueError, match="cannot be empty"):
        context_recall_score(
            ["context"],
            "",
        )


# Dataset evaluation and aggregate reporting


def test_evaluate_rag_dataset_returns_summary_and_per_example_metrics():
    dataset = [
        {
            "query": "What is Python?",
            "answer": "Python is a programming language",
            "contexts": ["Python is a programming language"],
            "reference_answer": "Python is a programming language",
            "retrieved_ids": ["doc1", "doc2"],
            "relevant_ids": ["doc1"],
        },
        {
            "query": "What is Java?",
            "answer": "Java is a programming language",
            "contexts": ["Java is a programming language"],
            "reference_answer": "Java is a programming language",
            "retrieved_ids": ["doc3"],
            "relevant_ids": ["doc3"],
        },
    ]

    report = evaluate_rag_dataset(
        dataset,
        k=2,
    )

    assert report["example_count"] == 2
    assert report["k"] == 2
    assert report["metrics"]["faithfulness"] == 1.0
    assert report["metrics"]["reference_answer_f1"] == 1.0

    # First example:
    # precision@2 = 1/2 = 0.5
    #
    # Second example:
    # precision@2 = 1/2 = 0.5
    #
    # Mean = 0.5.
    assert report["metrics"]["precision_at_k"] == pytest.approx(0.5)

    assert len(report["per_example"]) == 2
    assert "metric_notes" in report


def test_evaluate_rag_dataset_without_retrieval_labels():
    report = evaluate_rag_dataset(
        [
            {
                "query": "What is Python?",
                "answer": "Python is a language",
                "contexts": ["Python is a language"],
            }
        ]
    )

    assert report["example_count"] == 1
    assert "faithfulness" in report["metrics"]
    assert "precision_at_k" not in report["metrics"]


def test_evaluate_rag_dataset_rejects_empty_dataset():
    with pytest.raises(ValueError, match="cannot be empty"):
        evaluate_rag_dataset([])


def test_evaluate_rag_dataset_rejects_missing_required_fields():
    with pytest.raises(ValueError, match="missing required fields"):
        evaluate_rag_dataset(
            [
                {
                    "query": "Q",
                    "answer": "A",
                }
            ]
        )


def test_evaluate_rag_dataset_rejects_non_mapping_item():
    with pytest.raises(TypeError, match="must be a mapping"):
        evaluate_rag_dataset(["not a mapping"])


def test_evaluate_rag_dataset_requires_both_retrieval_label_fields():
    with pytest.raises(
        ValueError,
        match="both retrieved_ids and relevant_ids",
    ):
        evaluate_rag_dataset(
            [
                {
                    "query": "Q",
                    "answer": "A",
                    "contexts": ["A"],
                    "retrieved_ids": ["doc1"],
                }
            ]
        )


def test_evaluate_rag_dataset_rejects_non_positive_k():
    with pytest.raises(ValueError, match="greater than zero"):
        evaluate_rag_dataset(
            [
                {
                    "query": "Q",
                    "answer": "A",
                    "contexts": ["A"],
                }
            ],
            k=0,
        )