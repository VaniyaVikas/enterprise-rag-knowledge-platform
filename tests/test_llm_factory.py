
"""Tests for the Groq LLM factory."""

import pytest

from src import llm_factory
from src.llm_factory import DEFAULT_GROQ_MODEL, create_groq_llm


@pytest.fixture
def groq_environment(monkeypatch):
    """Configure a fake Groq API key for unit tests."""
    monkeypatch.setenv("GROQ_API_KEY", "test-groq-api-key")
    monkeypatch.delenv("GROQ_MODEL", raising=False)


def test_create_groq_llm_uses_default_model(
    monkeypatch,
    groq_environment,
):
    mock_llm = object()
    calls = []

    def fake_chat_groq(**kwargs):
        calls.append(kwargs)
        return mock_llm

    monkeypatch.setattr(llm_factory, "ChatGroq", fake_chat_groq)

    result = create_groq_llm()

    assert result is mock_llm
    assert calls == [
        {
            "model": DEFAULT_GROQ_MODEL,
            "api_key": "test-groq-api-key",
            "temperature": 0.0,
        }
    ]


def test_create_groq_llm_uses_environment_model(
    monkeypatch,
    groq_environment,
):
    monkeypatch.setenv("GROQ_MODEL", "custom-groq-model")
    monkeypatch.setattr(
        llm_factory,
        "ChatGroq",
        lambda **kwargs: kwargs,
    )

    result = create_groq_llm()

    assert result["model"] == "custom-groq-model"


def test_create_groq_llm_uses_explicit_model(
    monkeypatch,
    groq_environment,
):
    monkeypatch.setenv("GROQ_MODEL", "environment-model")
    monkeypatch.setattr(
        llm_factory,
        "ChatGroq",
        lambda **kwargs: kwargs,
    )

    result = create_groq_llm(model="explicit-model")

    assert result["model"] == "explicit-model"


def test_create_groq_llm_rejects_missing_api_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    with pytest.raises(ValueError, match="GROQ_API_KEY is missing"):
        create_groq_llm()


def test_create_groq_llm_rejects_blank_environment_model(
    monkeypatch,
    groq_environment,
):
    monkeypatch.setenv("GROQ_MODEL", "   ")

    with pytest.raises(ValueError, match="model name cannot be empty"):
        create_groq_llm()


@pytest.mark.parametrize(
    "temperature",
    [-0.1, 2.1, float("nan"), float("inf"), -float("inf"), True, "0.5", None],
)
def test_create_groq_llm_rejects_invalid_temperature(
    monkeypatch,
    groq_environment,
    temperature,
):
    with pytest.raises(ValueError, match="temperature must be"):
        create_groq_llm(temperature=temperature)


@pytest.mark.parametrize("temperature", [0, 0.5, 1, 2])
def test_create_groq_llm_accepts_valid_temperature(
    monkeypatch,
    groq_environment,
    temperature,
):
    monkeypatch.setattr(
        llm_factory,
        "ChatGroq",
        lambda **kwargs: kwargs,
    )

    result = create_groq_llm(temperature=temperature)

    assert result["temperature"] == temperature
    assert result["api_key"] == "test-groq-api-key"
