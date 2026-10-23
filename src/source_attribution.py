
"""Attach source citations to generated RAG answers."""


class SourceAttributor:
    """Build numbered citations and attach them to generated answers."""

    def __init__(self, include_document_text: bool = False) -> None:
        if not isinstance(include_document_text, bool):
            raise TypeError("include_document_text must be a boolean.")

        self.include_document_text = include_document_text

    @staticmethod
    def _validate_sources(sources: list[dict]) -> list[dict]:
        if not isinstance(sources, list):
            raise TypeError("sources must be a list.")

        validated = []

        for position, source in enumerate(sources, start=1):
            if not isinstance(source, dict):
                raise TypeError("each source must be a dictionary.")

            document = source.get("document")

            if not isinstance(document, str):
                raise TypeError("each source must contain string document text.")

            if not document.strip():
                raise ValueError("source document text cannot be empty.")

            metadata = source.get("metadata", {})

            if not isinstance(metadata, dict):
                raise TypeError("source metadata must be a dictionary.")

            validated.append(
                {
                    "citation_id": position,
                    "index": source.get("index"),
                    "document": document.strip(),
                    "metadata": metadata.copy(),
                }
            )

        return validated

    def format_citations(self, sources: list[dict]) -> list[dict]:
        """Return numbered citations for the supplied sources."""
        validated = self._validate_sources(sources)
        citations = []

        for source in validated:
            metadata = source["metadata"]

            citation = {
                "citation_id": source["citation_id"],
                "source": metadata.get(
                    "source",
                    metadata.get("url", "Unknown source"),
                ),
            }

            if source["index"] is not None:
                citation["index"] = source["index"]

            if self.include_document_text:
                citation["document"] = source["document"]

            citations.append(citation)

        return citations

    def attach_citations(
        self,
        answer: str,
        sources: list[dict],
    ) -> dict:
        """Attach formatted source citations to an answer."""
        if not isinstance(answer, str):
            raise TypeError("answer must be a string.")

        if not answer.strip():
            raise ValueError("answer cannot be empty.")

        citations = self.format_citations(sources)

        return {
            "answer": answer.strip(),
            "citations": citations,
        }
