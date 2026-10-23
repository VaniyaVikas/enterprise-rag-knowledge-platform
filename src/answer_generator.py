from collections.abc import Callable


class AnswerGenerator:
    """Generate answers from a query and retrieved document context."""

    def __init__(
        self,
        answer_generator: Callable[[str, str], str],
        no_context_answer: str = (
            "I could not find relevant information to answer this question."
        ),
    ) -> None:
        if not callable(answer_generator):
            raise TypeError("answer_generator must be callable.")

        if not isinstance(no_context_answer, str):
            raise TypeError("no_context_answer must be a string.")

        if not no_context_answer.strip():
            raise ValueError("no_context_answer cannot be empty.")

        self.answer_generator = answer_generator
        self.no_context_answer = no_context_answer.strip()

    @staticmethod
    def _validate_query(query: str) -> str:
        """Validate and normalize the user's query."""
        if not isinstance(query, str):
            raise TypeError("query must be a string.")

        if not query.strip():
            raise ValueError("query cannot be empty.")

        return query.strip()

    @staticmethod
    def _prepare_context(
        retrieved_documents: list[dict],
    ) -> tuple[str, list[dict]]:
        """Build context text and source metadata from retrieved documents."""
        if not isinstance(retrieved_documents, list):
            raise TypeError("retrieved_documents must be a list.")

        context_parts = []
        sources = []

        for position, result in enumerate(retrieved_documents, start=1):
            if not isinstance(result, dict):
                raise TypeError(
                    "each retrieved document must be a dictionary."
                )

            document = result.get("document")

            if not isinstance(document, str):
                raise TypeError(
                    "each retrieved document must contain string text."
                )

            if not document.strip():
                raise ValueError(
                    "retrieved document text cannot be empty."
                )

            metadata = result.get("metadata", {})

            if not isinstance(metadata, dict):
                raise TypeError("document metadata must be a dictionary.")

            context_parts.append(f"[{position}] {document.strip()}")

            sources.append(
                {
                    "index": result.get("index"),
                    "document": document.strip(),
                    "metadata": metadata.copy(),
                }
            )

        context = "\n\n".join(context_parts)

        return context, sources

    def generate(
        self,
        query: str,
        retrieved_documents: list[dict],
    ) -> dict:
        """Generate an answer and return its source documents."""
        query = self._validate_query(query)

        context, sources = self._prepare_context(retrieved_documents)

        if not sources:
            return {
                "query": query,
                "answer": self.no_context_answer,
                "sources": [],
            }

        answer = self.answer_generator(query, context)

        if not isinstance(answer, str):
            raise TypeError("answer generator must return a string.")

        if not answer.strip():
            raise ValueError("generated answer cannot be empty.")

        return {
            "query": query,
            "answer": answer.strip(),
            "sources": sources,
        }
