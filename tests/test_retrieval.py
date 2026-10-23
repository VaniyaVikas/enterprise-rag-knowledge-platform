from unittest.mock import MagicMock

import pytest
from langchain_huggingface import HuggingFaceEmbeddings

from src.retrieval import build_vector_store, retrieve_documents
from src.vector_store import VectorStore


@pytest.fixture
def mock_model():
    model = MagicMock(spec=HuggingFaceEmbeddings)

    model.embed_documents.return_value = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ]
    model.embed_query.return_value = [1.0, 0.0, 0.0]

    return model


@pytest.fixture
def documents():
    return [
        "Python is a programming language.",
        "Machine learning uses data.",
    ]


@pytest.fixture
def metadata():
    return [
        {"source": "python.txt"},
        {"source": "ml.txt"},
    ]


def test_build_vector_store_returns_vector_store(mock_model, documents):
    store = build_vector_store(documents, model=mock_model)

    assert isinstance(store, VectorStore)


def test_build_vector_store_stores_documents(mock_model, documents):
    store = build_vector_store(documents, model=mock_model)

    assert store.documents == documents


def test_build_vector_store_stores_metadata(mock_model, documents, metadata):
    store = build_vector_store(
        documents,
        model=mock_model,
        metadata=metadata,
    )

    assert store.metadata == metadata


def test_build_vector_store_uses_empty_metadata_by_default(mock_model, documents):
    store = build_vector_store(documents, model=mock_model)

    assert store.metadata == [{}, {}]


def test_build_vector_store_embeds_documents(mock_model, documents):
    build_vector_store(documents, model=mock_model)

    mock_model.embed_documents.assert_called_once_with(documents)


def test_build_vector_store_creates_model_when_not_provided(
    monkeypatch,
    documents,
):
    fake_model = MagicMock(spec=HuggingFaceEmbeddings)
    fake_model.embed_documents.return_value = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ]

    monkeypatch.setattr(
        "src.retrieval.create_embedding_model",
        lambda: fake_model,
    )

    store = build_vector_store(documents)

    assert isinstance(store, VectorStore)
    assert store.documents == documents


def test_build_vector_store_rejects_non_list_documents():
    with pytest.raises(TypeError, match="documents must be a list"):
        build_vector_store("not a list")


def test_build_vector_store_rejects_empty_documents():
    with pytest.raises(ValueError, match="documents cannot be empty"):
        build_vector_store([])


def test_build_vector_store_rejects_empty_document_text(mock_model):
    with pytest.raises(ValueError, match="non-empty strings"):
        build_vector_store(["Python", "   "], model=mock_model)


def test_build_vector_store_rejects_non_string_documents(mock_model):
    with pytest.raises(ValueError, match="non-empty strings"):
        build_vector_store(["Python", 123], model=mock_model)


def test_build_vector_store_rejects_non_list_metadata(mock_model, documents):
    with pytest.raises(TypeError, match="metadata must be a list"):
        build_vector_store(
            documents,
            model=mock_model,
            metadata="invalid",
        )


def test_build_vector_store_rejects_metadata_length_mismatch(
    mock_model,
    documents,
):
    with pytest.raises(ValueError, match="metadata must match"):
        build_vector_store(
            documents,
            model=mock_model,
            metadata=[{"source": "only-one.txt"}],
        )


def test_retrieve_documents_returns_ranked_results(mock_model, documents):
    store = build_vector_store(documents, model=mock_model)

    results = retrieve_documents(
        "Python",
        store,
        model=mock_model,
    )

    assert len(results) == 2
    assert results[0]["document"] == "Python is a programming language."
    assert results[0]["score"] >= results[1]["score"]


def test_retrieve_documents_passes_k_to_store(mock_model, documents):
    store = build_vector_store(documents, model=mock_model)

    results = retrieve_documents(
        "Python",
        store,
        model=mock_model,
        k=1,
    )

    assert len(results) == 1


def test_retrieve_documents_embeds_query(mock_model, documents):
    store = build_vector_store(documents, model=mock_model)

    retrieve_documents(
        "Python",
        store,
        model=mock_model,
    )

    mock_model.embed_query.assert_called_once_with("Python")


def test_retrieve_documents_uses_default_model(monkeypatch, mock_model, documents):
    store = build_vector_store(documents, model=mock_model)

    fake_model = MagicMock(spec=HuggingFaceEmbeddings)
    fake_model.embed_query.return_value = [1.0, 0.0, 0.0]

    monkeypatch.setattr(
        "src.retrieval.create_embedding_model",
        lambda: fake_model,
    )

    results = retrieve_documents("Python", store)

    fake_model.embed_query.assert_called_once_with("Python")
    assert results


def test_retrieve_documents_rejects_non_string_query(mock_model, documents):
    store = build_vector_store(documents, model=mock_model)

    with pytest.raises(TypeError, match="query must be a string"):
        retrieve_documents(123, store, model=mock_model)


def test_retrieve_documents_rejects_empty_query(mock_model, documents):
    store = build_vector_store(documents, model=mock_model)

    with pytest.raises(ValueError, match="query cannot be empty"):
        retrieve_documents("   ", store, model=mock_model)


def test_retrieve_documents_rejects_invalid_store():
    with pytest.raises(TypeError, match="store must be a VectorStore"):
        retrieve_documents("Python", "not a store")


@pytest.mark.parametrize("invalid_k", [0, -1])
def test_retrieve_documents_rejects_non_positive_k(
    mock_model,
    documents,
    invalid_k,
):
    store = build_vector_store(documents, model=mock_model)

    with pytest.raises(ValueError, match="k must be greater than zero"):
        retrieve_documents(
            "Python",
            store,
            model=mock_model,
            k=invalid_k,
        )


def test_retrieve_documents_rejects_non_integer_k(mock_model, documents):
    store = build_vector_store(documents, model=mock_model)

    with pytest.raises(TypeError, match="k must be an integer"):
        retrieve_documents(
            "Python",
            store,
            model=mock_model,
            k=1.5,
        )