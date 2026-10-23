
from collections.abc import Callable

from src.embeddings import (
    create_embedding_model,
    embed_texts,
    embed_query,
)
from src.vector_store import VectorStore


class HyDERetriever:
    """Retrieve real documents using a generated hypothetical document."""

    def __init__(
        self,
        documents: list[dict],
        hypothetical_document_generator: Callable[[str], str],
        model=None,
    ):
        if not isinstance(documents, list) or not documents:
            raise ValueError("documents must be a non-empty list")

        if not callable(hypothetical_document_generator):
            raise TypeError(
                "hypothetical_document_generator must be callable"
            )

        self.model = model if model is not None else create_embedding_model()
        self.hypothetical_document_generator = (
            hypothetical_document_generator
        )
        self.documents = []
        self.metadata = []

        for item in documents:
            if not isinstance(item, dict):
                raise TypeError("each document must be a dictionary")

            document = item.get("document")
            metadata = item.get("metadata", {})

            if not isinstance(document, str) or not document.strip():
                raise ValueError("document must be a non-empty string")

            if not isinstance(metadata, dict):
                raise TypeError("metadata must be a dictionary")

            self.documents.append(document)
            self.metadata.append(metadata)

        vectors = embed_texts(self.documents, self.model)

        self.vector_store = VectorStore(dimension=len(vectors[0]))
        self.vector_store.add(
            vectors=vectors,
            documents=self.documents,
            metadata=self.metadata,
        )

    def generate_hypothetical_document(self, query: str) -> str:
        """Generate and validate a hypothetical document for a query."""
        if not isinstance(query, str):
            raise TypeError("query must be a string")

        if not query.strip():
            raise ValueError("query must not be empty")

        hypothetical_document = self.hypothetical_document_generator(
            query
        )

        if (
            not isinstance(hypothetical_document, str)
            or not hypothetical_document.strip()
        ):
            raise ValueError(
                "hypothetical document must be a non-empty string"
            )

        return hypothetical_document.strip()

    def search(self, query: str, k: int = 3) -> list[dict]:
        """Search the vector store using the hypothetical document vector."""
        if not isinstance(query, str):
            raise TypeError("query must be a string")

        if not query.strip():
            raise ValueError("query must not be empty")

        if isinstance(k, bool) or not isinstance(k, int):
            raise TypeError("k must be an integer")

        if k <= 0:
            raise ValueError("k must be positive")

        hypothetical_document = self.generate_hypothetical_document(query)
        hypothetical_vector = embed_query(
            hypothetical_document,
            self.model,
        )

        return self.vector_store.search(
            hypothetical_vector,
            k=min(k, len(self.documents)),
        )
