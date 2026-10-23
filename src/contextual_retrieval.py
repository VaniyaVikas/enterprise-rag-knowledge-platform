
from collections.abc import Callable


class ContextualRetriever:
    """Add document-level context to chunks before retrieval."""

    def __init__(
        self,
        context_generator: Callable[[str, str], str],
    ):
        if not callable(context_generator):
            raise TypeError("context_generator must be callable")

        self.context_generator = context_generator

    def enrich_chunk(
        self,
        document: str,
        chunk: str,
    ) -> str:
        """Generate context for one chunk and combine it with the chunk."""
        if not isinstance(document, str):
            raise TypeError("document must be a string")

        if not document.strip():
            raise ValueError("document must not be empty")

        if not isinstance(chunk, str):
            raise TypeError("chunk must be a string")

        if not chunk.strip():
            raise ValueError("chunk must not be empty")

        context = self.context_generator(document, chunk)

        if not isinstance(context, str):
            raise TypeError("generated context must be a string")

        if not context.strip():
            raise ValueError("generated context must not be empty")

        return f"{context.strip()}\n\n{chunk.strip()}"

    def enrich_chunks(
        self,
        document: str,
        chunks: list[str],
    ) -> list[str]:
        """Enrich multiple chunks from the same source document."""
        if not isinstance(chunks, list):
            raise TypeError("chunks must be a list")

        if not chunks:
            raise ValueError("chunks must not be empty")

        return [
            self.enrich_chunk(document, chunk)
            for chunk in chunks
        ]
