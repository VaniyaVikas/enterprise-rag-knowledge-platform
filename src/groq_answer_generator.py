"""Groq-backed answer generation adapter for the RAG pipeline."""

from typing import Any

from src.llm_factory import create_groq_llm


def create_groq_answer_generator(
    model: str | None = None,
    temperature: float = 0.0,
):
    """Create a callable that generates answers using a Groq chat model."""
    llm = create_groq_llm(
        model=model,
        temperature=temperature,
    )

    def generate_answer(query: str, prompt: str) -> str:
        """Generate answer text from the prepared RAG prompt."""
        if not isinstance(query, str):
            raise TypeError("query must be a string.")

        if not query.strip():
            raise ValueError("query cannot be empty.")

        if not isinstance(prompt, str):
            raise TypeError("prompt must be a string.")

        if not prompt.strip():
            raise ValueError("prompt cannot be empty.")

        response: Any = llm.invoke(prompt)

        content = getattr(response, "content", None)

        if isinstance(content, str):
            answer = content
        elif isinstance(content, list):
            text_parts = [
                item["text"]
                for item in content
                if isinstance(item, dict)
                and isinstance(item.get("text"), str)
            ]
            answer = "\n".join(text_parts)
        else:
            raise TypeError(
                "Groq response content must be a string or a list of text blocks."
            )

        if not answer.strip():
            raise ValueError("Groq returned an empty answer.")

        return answer.strip()

    return generate_answer
