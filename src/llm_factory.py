
"""Factory for creating Groq chat language models."""

import os
from math import isfinite
from typing import Any

from langchain_groq import ChatGroq


DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"


def create_groq_llm(
    model: str | None = None,
    temperature: float = 0.0,
) -> Any:
    """Create a Groq chat model using environment-based configuration."""
    api_key = os.getenv("GROQ_API_KEY", "").strip()

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is missing. Set it in your environment or .env file."
        )

    model_name = model or os.getenv(
        "GROQ_MODEL",
        DEFAULT_GROQ_MODEL,
    )
    model_name = model_name.strip()

    if not model_name:
        raise ValueError("Groq model name cannot be empty.")

    if (
        isinstance(temperature, bool)
        or not isinstance(temperature, (int, float))
        or not isfinite(temperature)
        or not 0 <= temperature <= 2
    ):
        raise ValueError("temperature must be a finite number between 0 and 2.")

    return ChatGroq(
        model=model_name,
        api_key=api_key,
        temperature=temperature,
    )
