
from unittest.mock import MagicMock

import pytest
from langchain_huggingface import HuggingFaceEmbeddings

from src.hybrid_retriever import HybridRetriever
from src.retrieval import build_vector_store
from src.vector_store import VectorStore


@pytest.fixture
def documents():
    return [
        "Python programming and debugging",
        "Machine learning with neural networks",
        "Python error handling and exceptions",
        "Cooking recipes and ingredients",
    ]


@pytest.fixture
def mock_model():
    model = MagicMock(spec=HuggingFaceEmbeddings)

    model.embed_documents.return_value = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.8, 0.2, 0.0],
        [0.0, 0.0, 1.0],
    ]
    model.embed_query.return_value = [1.0, 0.0, 0.0]

    return model


@pytest.fixture
def vector_store(documents, mock_model):
    return build_vector_store(documents, model=mock_model)


@pytest.fixture
def hybrid_retriever(documents, vector_store, mock_model):
    return HybridRetriever(
        documents=documents,
        vector_store=vector_store,
        embedding_model=mock_model,
    )


def test_constructor_creates_bm25_retriever(hybrid_retriever):
    from src.bm25_retriever import BM25Retriever

    assert isinstance(hybrid_retriever.bm25_retriever, BM25Retriever)


def test_search_returns_list(hybrid_retriever):
    results = hybrid_retriever.search("Python debugging")

    assert isinstance(results, list)
    assert results


def test_search_respects_k(hybrid_retriever):
    results = hybrid_retriever.search("Python", k=2)

    assert len(results) == 2


def test_search_returns_all_documents_when_k_is_large(hybrid_retriever):
    results = hybrid_retriever.search("Python", k=10)

    assert len(results) == 4


def test_results_contain_expected_keys(hybrid_retriever):
    results = hybrid_retriever.search("Python")

    assert set(results[0]) == {
        "document",
        "metadata",
        "score",
        "index",
        "bm25_rank",
        "vector_rank",
    }


def test_results_have_valid_indices(hybrid_retriever):
    results = hybrid_retriever.search("Python")

    assert all(0 <= item["index"] < 4 for item in results)


def test_results_have_positive_rrf_scores(hybrid_retriever):
    results = hybrid_retriever.search("Python")

    assert all(item["score"] > 0 for item in results)


def test_results_are_sorted_by_rrf_score(hybrid_retriever):
    results = hybrid_retriever.search("Python", k=4)
    scores = [item["score"] for item in results]

    assert scores == sorted(scores, reverse=True)


def test_both_retrievers_contribute_to_results(hybrid_retriever):
    results = hybrid_retriever.search("Python")

    assert any(item["bm25_rank"] is not None for item in results)
    assert any(item["vector_rank"] is not None for item in results)


def test_search_uses_embedding_model(hybrid_retriever, mock_model):
    hybrid_retriever.search("Python")

    mock_model.embed_query.assert_called_once_with("Python")


def test_search_rejects_non_string_query(hybrid_retriever):
    with pytest.raises(TypeError, match="query must be a string"):
        hybrid_retriever.search(123)


def test_search_rejects_empty_query(hybrid_retriever):
    with pytest.raises(ValueError, match="query cannot be empty"):
        hybrid_retriever.search("   ")


@pytest.mark.parametrize("invalid_k", [0, -1])
def test_search_rejects_non_positive_k(hybrid_retriever, invalid_k):
    with pytest.raises(ValueError, match="k must be greater than zero"):
        hybrid_retriever.search("Python", k=invalid_k)


def test_search_rejects_non_integer_k(hybrid_retriever):
    with pytest.raises(TypeError, match="k must be an integer"):
        hybrid_retriever.search("Python", k=1.5)


def test_constructor_rejects_non_list_documents(vector_store):
    with pytest.raises(TypeError, match="documents must be a list"):
        HybridRetriever("invalid", vector_store)


def test_constructor_rejects_empty_documents(vector_store):
    with pytest.raises(ValueError, match="documents cannot be empty"):
        HybridRetriever([], vector_store)


def test_constructor_rejects_non_string_document(vector_store):
    with pytest.raises(ValueError, match="non-empty strings"):
        HybridRetriever(["Python", 123], vector_store)


def test_constructor_rejects_invalid_vector_store(documents):
    with pytest.raises(TypeError, match="vector_store must be a VectorStore"):
        HybridRetriever(documents, "invalid")


def test_constructor_rejects_mismatched_documents(documents, vector_store):
    different_documents = documents.copy()
    different_documents[0] = "Different document"

    with pytest.raises(ValueError, match="must match the vector store"):
        HybridRetriever(different_documents, vector_store)


@pytest.mark.parametrize("invalid_constant", [0, -1])
def test_constructor_rejects_non_positive_rrf_constant(
    documents,
    vector_store,
    invalid_constant,
):
    with pytest.raises(ValueError, match="rrf_constant must be greater than zero"):
        HybridRetriever(
            documents,
            vector_store,
            rrf_constant=invalid_constant,
        )


def test_constructor_rejects_non_integer_rrf_constant(
    documents,
    vector_store,
):
    with pytest.raises(TypeError, match="rrf_constant must be an integer"):
        HybridRetriever(
            documents,
            vector_store,
            rrf_constant=1.5,
        )


def test_results_preserve_metadata(hybrid_retriever):
    results = hybrid_retriever.search("Python")

    for result in results:
        assert isinstance(result["metadata"], dict)


def test_search_with_custom_rrf_constant(
    documents,
    vector_store,
    mock_model,
):
    retriever = HybridRetriever(
        documents,
        vector_store,
        embedding_model=mock_model,
        rrf_constant=10,
    )

    results = retriever.search("Python")

    assert results
    assert retriever.rrf_constant == 10
