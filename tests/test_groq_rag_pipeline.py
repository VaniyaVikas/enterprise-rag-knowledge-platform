"""Tests for the Groq-backed production RAG pipeline factory."""

import pytest

import src.groq_rag_pipeline as groq_rag_module
from src.production_rag_pipeline import ProductionRAGPipeline
from src.vector_store import VectorStore


@pytest.fixture
def store():
    return VectorStore(dimension=2)


def test_factory_creates_production_pipeline(store, monkeypatch):
    fake_generator = lambda query, prompt: "A grounded answer."
    calls = []

    def fake_create_generator(**kwargs):
        calls.append(kwargs)
        return fake_generator

    monkeypatch.setattr(
        groq_rag_module,
        "create_groq_answer_generator",
        fake_create_generator,
    )

    pipeline = groq_rag_module.create_groq_rag_pipeline(store)

    assert isinstance(pipeline, ProductionRAGPipeline)
    assert pipeline.store is store
    assert pipeline.answer_pipeline.answer_generator is fake_generator
    assert pipeline.k == 3
    assert pipeline.model is None
    assert calls == [{"model": None, "temperature": 0.0}]


def test_factory_passes_model_configuration(store, monkeypatch):
    fake_generator = lambda query, prompt: "Answer."
    calls = []

    def fake_create_generator(**kwargs):
        calls.append(kwargs)
        return fake_generator

    monkeypatch.setattr(
        groq_rag_module,
        "create_groq_answer_generator",
        fake_create_generator,
    )

    embedding_model = object()

    pipeline = groq_rag_module.create_groq_rag_pipeline(
        store=store,
        embedding_model=embedding_model,
        k=5,
        groq_model="test-groq-model",
        temperature=0.4,
    )

    assert pipeline.model is embedding_model
    assert pipeline.k == 5
    assert calls == [
        {"model": "test-groq-model", "temperature": 0.4}
    ]


def test_factory_rejects_invalid_store():
    with pytest.raises(TypeError, match="VectorStore"):
        groq_rag_module.create_groq_rag_pipeline("invalid")


def test_factory_propagates_generator_configuration_error(
    store,
    monkeypatch,
):
    def fail_to_create_generator(**kwargs):
        raise ValueError("test configuration error")

    monkeypatch.setattr(
        groq_rag_module,
        "create_groq_answer_generator",
        fail_to_create_generator,
    )

    with pytest.raises(ValueError, match="test configuration error"):
        groq_rag_module.create_groq_rag_pipeline(store)
