"""Tests for the Groq answer-generation adapter."""

from types import SimpleNamespace

import pytest

from src import groq_answer_generator


class FakeChatModel:
    """Minimal fake model that records prompts and returns a response."""

    def __init__(self, response):
        self.response = response
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return self.response


def test_groq_answer_generator_returns_response_text(monkeypatch):
    model = FakeChatModel(
        SimpleNamespace(content="  Answer from Groq.  ")
    )
    monkeypatch.setattr(
        groq_answer_generator,
        "create_groq_llm",
        lambda **kwargs: model,
    )

    generate = groq_answer_generator.create_groq_answer_generator()

    result = generate("What is RAG?", "Use only this context.")

    assert result == "Answer from Groq."
    assert model.prompts == ["Use only this context."]


def test_groq_answer_generator_passes_model_configuration(monkeypatch):
    model = FakeChatModel(SimpleNamespace(content="Answer."))
    calls = []

    def fake_factory(**kwargs):
        calls.append(kwargs)
        return model

    monkeypatch.setattr(
        groq_answer_generator,
        "create_groq_llm",
        fake_factory,
    )

    groq_answer_generator.create_groq_answer_generator(
        model="test-model",
        temperature=0.4,
    )

    assert calls == [{"model": "test-model", "temperature": 0.4}]


def test_groq_answer_generator_supports_text_blocks(monkeypatch):
    model = FakeChatModel(
        SimpleNamespace(
            content=[
                {"type": "text", "text": "First part."},
                {"type": "text", "text": "Second part."},
            ]
        )
    )
    monkeypatch.setattr(
        groq_answer_generator,
        "create_groq_llm",
        lambda **kwargs: model,
    )

    generate = groq_answer_generator.create_groq_answer_generator()

    assert generate("Question", "Context") == "First part.\nSecond part."


@pytest.mark.parametrize(
    ("query", "prompt", "error", "message"),
    [
        (None, "Context", TypeError, "query must be a string"),
        ("  ", "Context", ValueError, "query cannot be empty"),
        ("Question", None, TypeError, "prompt must be a string"),
        ("Question", "  ", ValueError, "prompt cannot be empty"),
    ],
)
def test_groq_answer_generator_validates_inputs(
    monkeypatch,
    query,
    prompt,
    error,
    message,
):
    model = FakeChatModel(SimpleNamespace(content="Answer."))
    monkeypatch.setattr(
        groq_answer_generator,
        "create_groq_llm",
        lambda **kwargs: model,
    )

    generate = groq_answer_generator.create_groq_answer_generator()

    with pytest.raises(error, match=message):
        generate(query, prompt)


def test_groq_answer_generator_rejects_unexpected_response(monkeypatch):
    model = FakeChatModel(SimpleNamespace(content=None))
    monkeypatch.setattr(
        groq_answer_generator,
        "create_groq_llm",
        lambda **kwargs: model,
    )

    generate = groq_answer_generator.create_groq_answer_generator()

    with pytest.raises(TypeError, match="response content"):
        generate("Question", "Context")


def test_groq_answer_generator_rejects_empty_response(monkeypatch):
    model = FakeChatModel(SimpleNamespace(content="   "))
    monkeypatch.setattr(
        groq_answer_generator,
        "create_groq_llm",
        lambda **kwargs: model,
    )

    generate = groq_answer_generator.create_groq_answer_generator()

    with pytest.raises(ValueError, match="empty answer"):
        generate("Question", "Context")
