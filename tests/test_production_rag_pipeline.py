
"""Tests for the production RAG pipeline."""

import pytest

import src.production_rag_pipeline as production_module
from src.answer_pipeline import AnswerPipeline
from src.production_rag_pipeline import ProductionRAGPipeline
from src.vector_store import VectorStore


class FakeAnswerPipeline:
    """Provide a lightweight answer pipeline for orchestration tests."""

    def __init__(self):
        self.last_query = None
        self.last_documents = None

    def run(self, query, retrieved_documents):
        self.last_query = query
        self.last_documents = retrieved_documents

        return {
            "query": query,
            "answer": "The answer is grounded in the retrieved context.",
            "is_valid": True,
            "reason": None,
            "action": "accept",
            "grounding": {
                "grounding_score": 1.0,
                "is_grounded": True,
                "sentences": [],
            },
            "citations": [],
        }


@pytest.fixture
def store():
    """Create a small valid vector store."""
    return VectorStore(dimension=2)


@pytest.fixture
def documents():
    """Return sample retrieval results."""
    return [
        {
            "index": 0,
            "document": "RAG combines retrieval with answer generation.",
            "metadata": {"source": "guide.txt"},
        },
        {
            "index": 1,
            "document": "Retrieved context helps ground an answer.",
            "metadata": {"source": "notes.txt"},
        },
    ]


@pytest.fixture
def fake_answer_pipeline():
    return FakeAnswerPipeline()


@pytest.fixture
def pipeline(store, fake_answer_pipeline):
    return ProductionRAGPipeline(
        store=store,
        answer_pipeline=fake_answer_pipeline,
        k=2,
    )


def test_constructor_rejects_invalid_store():
    with pytest.raises(TypeError, match="VectorStore"):
        ProductionRAGPipeline(
            store="not a vector store",
            answer_pipeline=FakeAnswerPipeline(),
        )


@pytest.mark.parametrize("k", [0, -1])
def test_constructor_rejects_non_positive_k(store, k):
    with pytest.raises(ValueError, match="greater than zero"):
        ProductionRAGPipeline(
            store=store,
            answer_pipeline=FakeAnswerPipeline(),
            k=k,
        )


@pytest.mark.parametrize("k", [True, 2.5, "3"])
def test_constructor_rejects_non_integer_k(store, k):
    with pytest.raises(TypeError, match="integer"):
        ProductionRAGPipeline(
            store=store,
            answer_pipeline=FakeAnswerPipeline(),
            k=k,
        )


def test_constructor_requires_answer_generator_or_pipeline(store):
    with pytest.raises(TypeError, match="answer_generator"):
        ProductionRAGPipeline(store=store)


def test_constructor_rejects_non_callable_answer_generator(store):
    with pytest.raises(TypeError, match="answer_generator"):
        ProductionRAGPipeline(
            store=store,
            answer_generator="not callable",
        )


def test_constructor_rejects_pipeline_without_callable_run(store):
    with pytest.raises(TypeError, match="callable run"):
        ProductionRAGPipeline(
            store=store,
            answer_pipeline=object(),
        )


def test_constructor_builds_existing_answer_pipeline(store):
    pipeline = ProductionRAGPipeline(
        store=store,
        answer_generator=lambda query, prompt: "A valid answer.",
    )

    assert isinstance(pipeline.answer_pipeline, AnswerPipeline)


@pytest.mark.parametrize("query", [None, 123, ["question"]])
def test_run_rejects_non_string_query(pipeline, query):
    with pytest.raises(TypeError, match="query must be a string"):
        pipeline.run(query)


@pytest.mark.parametrize("query", ["", "   ", "\n\t"])
def test_run_rejects_empty_query(pipeline, query):
    with pytest.raises(ValueError, match="query cannot be empty"):
        pipeline.run(query)


def test_run_retrieves_documents_and_returns_pipeline_response(
    pipeline,
    store,
    documents,
    fake_answer_pipeline,
    monkeypatch,
):
    calls = {}

    def fake_retrieve_documents(**kwargs):
        calls.update(kwargs)
        return documents

    monkeypatch.setattr(
        production_module,
        "retrieve_documents",
        fake_retrieve_documents,
    )

    result = pipeline.run("  What is RAG?  ")

    assert calls["query"] == "What is RAG?"
    assert calls["store"] is store
    assert calls["model"] is None
    assert calls["k"] == 2

    assert fake_answer_pipeline.last_query == "What is RAG?"
    assert fake_answer_pipeline.last_documents == documents

    assert result["answer"] == (
        "The answer is grounded in the retrieved context."
    )
    assert result["is_valid"] is True
    assert result["retrieval"] == {
        "requested_k": 2,
        "document_count": 2,
    }


def test_run_passes_configured_model_to_retrieval(
    store,
    documents,
    monkeypatch,
):
    model = object()
    fake_answer_pipeline = FakeAnswerPipeline()
    pipeline = ProductionRAGPipeline(
        store=store,
        answer_pipeline=fake_answer_pipeline,
        model=model,
        k=4,
    )
    calls = {}

    def fake_retrieve_documents(**kwargs):
        calls.update(kwargs)
        return documents

    monkeypatch.setattr(
        production_module,
        "retrieve_documents",
        fake_retrieve_documents,
    )

    result = pipeline.run("Explain RAG")

    assert calls["model"] is model
    assert calls["k"] == 4
    assert result["retrieval"]["requested_k"] == 4


def test_run_supports_empty_retrieval_results(
    pipeline,
    fake_answer_pipeline,
    monkeypatch,
):
    monkeypatch.setattr(
        production_module,
        "retrieve_documents",
        lambda **kwargs: [],
    )

    result = pipeline.run("A question with no matching documents")

    assert fake_answer_pipeline.last_documents == []
    assert result["retrieval"]["document_count"] == 0


def test_run_rejects_non_list_retrieval_result(
    pipeline,
    monkeypatch,
):
    monkeypatch.setattr(
        production_module,
        "retrieve_documents",
        lambda **kwargs: None,
    )

    with pytest.raises(TypeError, match="list of documents"):
        pipeline.run("A question")


def test_run_rejects_non_dictionary_pipeline_result(
    store,
    monkeypatch,
):
    class InvalidAnswerPipeline:
        def run(self, query, retrieved_documents):
            return "not a dictionary"

    pipeline = ProductionRAGPipeline(
        store=store,
        answer_pipeline=InvalidAnswerPipeline(),
    )

    monkeypatch.setattr(
        production_module,
        "retrieve_documents",
        lambda **kwargs: [],
    )

    with pytest.raises(TypeError, match="return a dictionary"):
        pipeline.run("A question")


def test_run_preserves_original_response_fields(
    pipeline,
    monkeypatch,
):
    original = {
        "query": "A question",
        "answer": "A safe fallback.",
        "is_valid": True,
        "reason": "no_context",
        "action": "fallback",
        "grounding": {
            "grounding_score": 0.0,
            "is_grounded": False,
            "sentences": [],
        },
        "citations": [],
    }

    class FallbackAnswerPipeline:
        def run(self, query, retrieved_documents):
            return original

    pipeline.answer_pipeline = FallbackAnswerPipeline()

    monkeypatch.setattr(
        production_module,
        "retrieve_documents",
        lambda **kwargs: [],
    )

    result = pipeline.run("A question")

    assert result["answer"] == "A safe fallback."
    assert result["action"] == "fallback"
    assert result["reason"] == "no_context"
    assert result["grounding"] == original["grounding"]
    assert result["citations"] == []
    assert "retrieval" not in original
    assert result["retrieval"]["document_count"] == 0
