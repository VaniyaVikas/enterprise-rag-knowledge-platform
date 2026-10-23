
from unittest.mock import MagicMock

import pytest
from langchain_huggingface import HuggingFaceEmbeddings

from src.embeddings import (
    DEFAULT_EMBEDDING_MODEL,
    create_embedding_model,
    embed_query,
    embed_texts,
)


def test_default_embedding_model_name():
    assert DEFAULT_EMBEDDING_MODEL == (
        "sentence-transformers/all-MiniLM-L6-v2"
    )


def test_create_embedding_model_uses_expected_configuration(monkeypatch):
    mock_model = MagicMock(spec=HuggingFaceEmbeddings)
    captured = {}

    def fake_embeddings(**kwargs):
        captured.update(kwargs)
        return mock_model

    monkeypatch.setattr(
        "src.embeddings.HuggingFaceEmbeddings",
        fake_embeddings,
    )

    result = create_embedding_model()

    assert result is mock_model
    assert captured["model_name"] == DEFAULT_EMBEDDING_MODEL
    assert captured["model_kwargs"] == {"device": "cpu"}
    assert captured["encode_kwargs"] == {"normalize_embeddings": True}


def test_create_embedding_model_accepts_custom_model_name(monkeypatch):
    captured = {}

    def fake_embeddings(**kwargs):
        captured.update(kwargs)
        return MagicMock(spec=HuggingFaceEmbeddings)

    monkeypatch.setattr(
        "src.embeddings.HuggingFaceEmbeddings",
        fake_embeddings,
    )

    create_embedding_model("organization/custom-model")

    assert captured["model_name"] == "organization/custom-model"


def test_create_embedding_model_rejects_non_string_name():
    with pytest.raises(TypeError, match="model_name must be a string."):
        create_embedding_model(123)


@pytest.mark.parametrize("model_name", ["", "   "])
def test_create_embedding_model_rejects_empty_name(model_name):
    with pytest.raises(ValueError, match="model_name cannot be empty."):
        create_embedding_model(model_name)


def test_embed_texts_returns_document_embeddings():
    model = MagicMock(spec=HuggingFaceEmbeddings)
    expected = [[0.1, 0.2], [0.3, 0.4]]
    model.embed_documents.return_value = expected

    result = embed_texts(["First document", "Second document"], model)

    assert result == expected
    model.embed_documents.assert_called_once_with(
        ["First document", "Second document"]
    )


def test_embed_texts_returns_empty_list_without_calling_model():
    model = MagicMock(spec=HuggingFaceEmbeddings)

    assert embed_texts([], model) == []
    model.embed_documents.assert_not_called()


def test_embed_texts_rejects_non_list_input():
    model = MagicMock(spec=HuggingFaceEmbeddings)

    with pytest.raises(TypeError, match="texts must be a list of strings."):
        embed_texts("single document", model)


def test_embed_texts_rejects_non_string_items():
    model = MagicMock(spec=HuggingFaceEmbeddings)

    with pytest.raises(TypeError, match="all texts must be strings."):
        embed_texts(["valid text", 123], model)


def test_embed_texts_rejects_empty_strings():
    model = MagicMock(spec=HuggingFaceEmbeddings)

    with pytest.raises(ValueError, match="texts cannot contain empty strings."):
        embed_texts(["valid text", "   "], model)


def test_embed_texts_rejects_invalid_model():
    with pytest.raises(
        TypeError,
        match="model must be a HuggingFaceEmbeddings instance.",
    ):
        embed_texts(["Some text"], object())


def test_embed_query_returns_query_embedding():
    model = MagicMock(spec=HuggingFaceEmbeddings)
    model.embed_query.return_value = [0.1, 0.2, 0.3]

    result = embed_query("What is RAG?", model)

    assert result == [0.1, 0.2, 0.3]
    model.embed_query.assert_called_once_with("What is RAG?")


@pytest.mark.parametrize("query", ["", "   "])
def test_embed_query_rejects_empty_query(query):
    model = MagicMock(spec=HuggingFaceEmbeddings)

    with pytest.raises(ValueError, match="query cannot be empty."):
        embed_query(query, model)


def test_embed_query_rejects_non_string_query():
    model = MagicMock(spec=HuggingFaceEmbeddings)

    with pytest.raises(TypeError, match="query must be a string."):
        embed_query(123, model)


def test_embed_query_rejects_invalid_model():
    with pytest.raises(
        TypeError,
        match="model must be a HuggingFaceEmbeddings instance.",
    ):
        embed_query("What is RAG?", object())
