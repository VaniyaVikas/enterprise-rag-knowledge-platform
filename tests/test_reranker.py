
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.reranker import Reranker


@pytest.fixture
def documents():
    return [
        {
            "document": "Python is a programming language.",
            "metadata": {"source": "python.txt"},
            "score": 0.80,
            "index": 0,
        },
        {
            "document": "Machine learning uses data to train models.",
            "metadata": {"source": "ml.txt"},
            "score": 0.70,
            "index": 1,
        },
        {
            "document": "Python supports functions and classes.",
            "metadata": {"source": "functions.txt"},
            "score": 0.75,
            "index": 2,
        },
    ]


@pytest.fixture
def mock_model():
    model = MagicMock()
    model.predict.return_value = np.array([0.4, 0.9, 0.7])
    return model


@pytest.fixture
def reranker(mock_model):
    return Reranker(model=mock_model)


def test_rerank_returns_list(reranker, documents):
    results = reranker.rerank("Python programming", documents)

    assert isinstance(results, list)


def test_rerank_sorts_by_relevance_score(reranker, documents):
    results = reranker.rerank("Python programming", documents)

    assert [
        item["rerank_score"] for item in results
    ] == [0.9, 0.7, 0.4]


def test_rerank_returns_highest_scoring_document_first(
    reranker,
    documents,
):
    results = reranker.rerank("Python programming", documents)

    assert results[0]["document"] == (
        "Machine learning uses data to train models."
    )


def test_rerank_respects_top_n(reranker, documents):
    results = reranker.rerank(
        "Python programming",
        documents,
        top_n=2,
    )

    assert len(results) == 2


def test_rerank_preserves_original_fields(reranker, documents):
    results = reranker.rerank("Python programming", documents)

    highest = results[0]

    assert highest["metadata"] == {"source": "ml.txt"}
    assert highest["score"] == 0.70
    assert highest["index"] == 1
    assert highest["rerank_score"] == pytest.approx(0.9)


def test_rerank_does_not_modify_original_documents(
    reranker,
    documents,
):
    original = [item.copy() for item in documents]

    reranker.rerank("Python programming", documents)

    assert documents == original


def test_rerank_passes_query_document_pairs_to_model(
    reranker,
    documents,
):
    reranker.rerank("Python programming", documents)

    reranker.model.predict.assert_called_once_with(
        [
            (
                "Python programming",
                "Python is a programming language.",
            ),
            (
                "Python programming",
                "Machine learning uses data to train models.",
            ),
            (
                "Python programming",
                "Python supports functions and classes.",
            ),
        ]
    )


def test_rerank_rejects_non_string_query(reranker, documents):
    with pytest.raises(TypeError, match="query must be a string"):
        reranker.rerank(123, documents)


def test_rerank_rejects_empty_query(reranker, documents):
    with pytest.raises(ValueError, match="query cannot be empty"):
        reranker.rerank("   ", documents)


def test_rerank_rejects_non_list_documents(reranker):
    with pytest.raises(TypeError, match="documents must be a list"):
        reranker.rerank("Python", "not a list")


def test_rerank_returns_empty_list_for_empty_documents(reranker):
    results = reranker.rerank("Python", [])

    assert results == []
    reranker.model.predict.assert_not_called()


def test_rerank_rejects_invalid_document_item(reranker):
    with pytest.raises(
        TypeError,
        match=r"documents\[0\] must be a dictionary",
    ):
        reranker.rerank("Python", ["not a dictionary"])


def test_rerank_rejects_missing_document_text(reranker):
    with pytest.raises(
        TypeError,
        match=r"documents\[0\]\['document'\] must be a string",
    ):
        reranker.rerank("Python", [{"metadata": {}}])


def test_rerank_rejects_empty_document_text(reranker):
    with pytest.raises(
        ValueError,
        match=r"documents\[0\]\['document'\] cannot be empty",
    ):
        reranker.rerank("Python", [{"document": "   "}])


@pytest.mark.parametrize("invalid_top_n", [0, -1])
def test_rerank_rejects_non_positive_top_n(
    reranker,
    documents,
    invalid_top_n,
):
    with pytest.raises(
        ValueError,
        match="top_n must be greater than zero",
    ):
        reranker.rerank(
            "Python",
            documents,
            top_n=invalid_top_n,
        )


@pytest.mark.parametrize("invalid_top_n", [1.5, "2", True])
def test_rerank_rejects_invalid_top_n_type(
    reranker,
    documents,
    invalid_top_n,
):
    with pytest.raises(
        TypeError,
        match="top_n must be an integer or None",
    ):
        reranker.rerank(
            "Python",
            documents,
            top_n=invalid_top_n,
        )


def test_constructor_rejects_model_without_predict():
    with pytest.raises(
        TypeError,
        match="model must provide a callable predict method",
    ):
        Reranker(model=object())


def test_constructor_rejects_non_string_model_name():
    with pytest.raises(TypeError, match="model_name must be a string"):
        Reranker(model_name=123)


def test_constructor_rejects_empty_model_name():
    with pytest.raises(ValueError, match="model_name cannot be empty"):
        Reranker(model_name="   ")


def test_rerank_rejects_wrong_number_of_model_scores(
    reranker,
    documents,
):
    reranker.model.predict.return_value = np.array([0.8])

    with pytest.raises(
        ValueError,
        match="exactly one score per document",
    ):
        reranker.rerank("Python programming", documents)
