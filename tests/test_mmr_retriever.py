from unittest.mock import MagicMock

import numpy as np
import pytest
from langchain_huggingface import HuggingFaceEmbeddings

from src.mmr_retriever import MMRRetriever
from src.retrieval import build_vector_store


@pytest.fixture
def documents():
    return [
        "Python programming and debugging",
        "Python error handling and exceptions",
        "Python functions and reusable code",
        "Machine learning with neural networks",
    ]


@pytest.fixture
def mock_model():
    model = MagicMock(spec=HuggingFaceEmbeddings)
    model.embed_documents.return_value = [
        [1.0, 0.0, 0.0],
        [0.9, 0.1, 0.0],
        [0.8, 0.2, 0.0],
        [0.0, 0.0, 1.0],
    ]
    model.embed_query.return_value = [1.0, 0.0, 0.0]
    return model


@pytest.fixture
def vector_store(documents, mock_model):
    return build_vector_store(documents, model=mock_model)


@pytest.fixture
def retriever(vector_store, mock_model):
    return MMRRetriever(
        vector_store=vector_store,
        embedding_model=mock_model,
    )


def test_search_returns_list(retriever):
    results = retriever.search("Python", k=2, fetch_k=4)

    assert isinstance(results, list)
    assert results


def test_search_respects_k(retriever):
    results = retriever.search("Python", k=2, fetch_k=4)

    assert len(results) <= 2


def test_search_returns_expected_fields(retriever):
    results = retriever.search("Python", k=2, fetch_k=4)

    assert results
    for item in results:
        assert "document" in item
        assert "metadata" in item
        assert "score" in item
        assert "index" in item
        assert "vector" not in item


def test_search_does_not_return_duplicate_documents(retriever):
    results = retriever.search("Python", k=4, fetch_k=4)
    indices = [item["index"] for item in results]

    assert len(indices) == len(set(indices))


def test_search_rejects_non_string_query(retriever):
    with pytest.raises(TypeError, match="query must be a string"):
        retriever.search(123)


def test_search_rejects_empty_query(retriever):
    with pytest.raises(ValueError, match="query cannot be empty"):
        retriever.search("   ")


@pytest.mark.parametrize("invalid_k", [0, -1])
def test_search_rejects_non_positive_k(retriever, invalid_k):
    with pytest.raises(ValueError, match="k must be greater than zero"):
        retriever.search("Python", k=invalid_k)


def test_search_rejects_non_integer_k(retriever):
    with pytest.raises(TypeError, match="k must be an integer"):
        retriever.search("Python", k=1.5)


@pytest.mark.parametrize("invalid_fetch_k", [0, -1])
def test_search_rejects_non_positive_fetch_k(retriever, invalid_fetch_k):
    with pytest.raises(
        ValueError,
        match="fetch_k must be greater than zero",
    ):
        retriever.search("Python", k=1, fetch_k=invalid_fetch_k)


def test_search_rejects_non_integer_fetch_k(retriever):
    with pytest.raises(TypeError, match="fetch_k must be an integer"):
        retriever.search("Python", k=1, fetch_k=1.5)


def test_search_rejects_fetch_k_smaller_than_k(retriever):
    with pytest.raises(
        ValueError,
        match="fetch_k must be greater than or equal to k",
    ):
        retriever.search("Python", k=3, fetch_k=2)


def test_constructor_rejects_invalid_store():
    with pytest.raises(TypeError, match="vector_store must be a VectorStore"):
        MMRRetriever("invalid")


def test_constructor_rejects_empty_store():
    from src.vector_store import VectorStore

    store = VectorStore(dimension=3)

    with pytest.raises(ValueError, match="must contain documents"):
        MMRRetriever(store)


@pytest.mark.parametrize("invalid_lambda", [-0.1, 1.1])
def test_constructor_rejects_lambda_out_of_range(invalid_lambda):
    from src.vector_store import VectorStore

    store = VectorStore(dimension=3)
    store.documents.append("Python")
    store.metadata.append({})

    with pytest.raises(ValueError, match="lambda_mult must be between 0 and 1"):
        MMRRetriever(store, lambda_mult=invalid_lambda)


def test_constructor_rejects_non_numeric_lambda():
    from src.vector_store import VectorStore

    store = VectorStore(dimension=3)
    store.documents.append("Python")
    store.metadata.append({})

    with pytest.raises(TypeError, match="lambda_mult must be a number"):
        MMRRetriever(store, lambda_mult="0.5")


def test_cosine_similarity_handles_zero_vectors():
    assert MMRRetriever._cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0


def test_cosine_similarity_identical_vectors():
    score = MMRRetriever._cosine_similarity([1.0, 0.0], [1.0, 0.0])

    assert np.isclose(score, 1.0)


def test_search_returns_empty_list_for_empty_search_results(retriever, monkeypatch):
    monkeypatch.setattr(
        retriever.vector_store,
        "search",
        lambda query_vector, k: [],
    )

    results = retriever.search("Python", k=2, fetch_k=4)

    assert results == []
