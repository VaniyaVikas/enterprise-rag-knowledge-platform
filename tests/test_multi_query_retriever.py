from unittest.mock import MagicMock

import pytest
from langchain_huggingface import HuggingFaceEmbeddings

from src.multi_query_retriever import MultiQueryRetriever
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
def retriever(vector_store, mock_model):
    return MultiQueryRetriever(
        vector_store=vector_store,
        embedding_model=mock_model,
    )


def test_generate_queries_returns_four_variations():
    queries = MultiQueryRetriever.generate_queries("Python errors")
    assert len(queries) == 4
    assert queries[0] == "Python errors"


def test_generate_queries_removes_duplicates():
    queries = MultiQueryRetriever.generate_queries("Python")
    normalized = [query.casefold() for query in queries]
    assert len(normalized) == len(set(normalized))


def test_generate_queries_rejects_non_string():
    with pytest.raises(TypeError, match="query must be a string"):
        MultiQueryRetriever.generate_queries(123)


def test_generate_queries_rejects_empty_query():
    with pytest.raises(ValueError, match="query cannot be empty"):
        MultiQueryRetriever.generate_queries("   ")


def test_search_returns_list(retriever):
    results = retriever.search("Python debugging")
    assert isinstance(results, list)
    assert results


def test_search_respects_k(retriever):
    results = retriever.search("Python", k=2)
    assert len(results) <= 2


def test_search_results_have_matched_queries(retriever):
    results = retriever.search("Python")
    assert all("matched_queries" in item for item in results)
    assert all(item["matched_queries"] for item in results)


def test_search_removes_duplicate_documents(retriever):
    results = retriever.search("Python", k=4)
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


def test_constructor_rejects_invalid_store():
    with pytest.raises(TypeError, match="vector_store must be a VectorStore"):
        MultiQueryRetriever("invalid")


def test_constructor_rejects_empty_store():
    store = VectorStore(dimension=3)
    with pytest.raises(ValueError, match="must contain documents"):
        MultiQueryRetriever(store)


def test_search_records_multiple_matching_queries(retriever):
    results = retriever.search("Python", k=4)
    assert results
    assert all(
        isinstance(item["matched_queries"], list)
        for item in results
    )

