
"""Build clean, numbered context from retrieved documents."""


class ContextBuilder:
    """Convert retrieved document results into LLM-ready context."""

    def __init__(
        self,
        max_documents: int | None = None,
        max_characters: int | None = None,
        include_metadata: bool = False,
    ) -> None:
        if max_documents is not None:
            if (
                isinstance(max_documents, bool)
                or not isinstance(max_documents, int)
            ):
                raise TypeError("max_documents must be an integer or None.")

            if max_documents <= 0:
                raise ValueError("max_documents must be positive.")

        if max_characters is not None:
            if (
                isinstance(max_characters, bool)
                or not isinstance(max_characters, int)
            ):
                raise TypeError(
                    "max_characters must be an integer or None."
                )

            if max_characters <= 0:
                raise ValueError("max_characters must be positive.")

        if not isinstance(include_metadata, bool):
            raise TypeError("include_metadata must be a boolean.")

        self.max_documents = max_documents
        self.max_characters = max_characters
        self.include_metadata = include_metadata

    @staticmethod
    def _validate_documents(
        documents: list[dict],
    ) -> list[dict]:
        if not isinstance(documents, list):
            raise TypeError("documents must be a list.")

        validated = []

        for document in documents:
            if not isinstance(document, dict):
                raise TypeError("each document must be a dictionary.")

            text = document.get("document")

            if not isinstance(text, str):
                raise TypeError(
                    "each document must contain string text."
                )

            if not text.strip():
                raise ValueError("document text cannot be empty.")

            metadata = document.get("metadata", {})

            if not isinstance(metadata, dict):
                raise TypeError("document metadata must be a dictionary.")

            validated.append(
                {
                    "document": text.strip(),
                    "metadata": metadata.copy(),
                    "index": document.get("index"),
                }
            )

        return validated

    def build(self, documents: list[dict]) -> str:
        """Build numbered context from retrieved documents."""
        validated = self._validate_documents(documents)

        if self.max_documents is not None:
            validated = validated[: self.max_documents]

        context_parts = []
        total_characters = 0

        for position, document in enumerate(validated, start=1):
            text = document["document"]

            if self.include_metadata and document["metadata"]:
                metadata_text = "; ".join(
                    f"{key}: {value}"
                    for key, value in document["metadata"].items()
                )
                text = f"Metadata: {metadata_text}\nContent: {text}"

            section = f"[{position}] {text}"

            if self.max_characters is not None:
                remaining = self.max_characters - total_characters

                if remaining <= 0:
                    break

                if len(section) > remaining:
                    section = section[:remaining]

            context_parts.append(section)
            total_characters += len(section)

            if (
                self.max_characters is not None
                and total_characters >= self.max_characters
            ):
                break

        return "\n\n".join(context_parts)
