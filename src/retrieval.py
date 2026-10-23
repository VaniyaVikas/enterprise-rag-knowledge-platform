
"""Build vector stores and retrieve relevant documents."""

from src.embeddings import (
    create_embedding_model,
    embed_query,
    embed_texts,
)
from src.vector_store import VectorStore


def build_vector_store(
    documents: list[str],
    model=None,
    metadata: list[dict] | None = None,
) -> VectorStore:
    """Create an in-memory FAISS store from document text."""
    if not isinstance(documents, list):
        raise TypeError("documents must be a list.")

    if not documents:
        raise ValueError("documents cannot be empty.")

    if not all(
        isinstance(document, str) and document.strip()
        for document in documents
    ):
        raise ValueError("documents must contain non-empty strings.")

    if metadata is not None:
        if not isinstance(metadata, list):
            raise TypeError("metadata must be a list.")

        if len(metadata) != len(documents):
            raise ValueError("metadata must match the number of documents.")

    if model is None:
        model = create_embedding_model()

    vectors = embed_texts(documents, model)

    if not vectors:
        raise ValueError("embedding model returned no vectors.")

    store = VectorStore(dimension=len(vectors[0]))
    store.add(
        vectors=vectors,
        documents=documents,
        metadata=metadata,
    )

    return store


def retrieve_documents(
    query: str,
    store: VectorStore,
    model=None,
    k: int = 3,
) -> list[dict]:
    """Return the top-k documents relevant to a query."""
    if not isinstance(query, str):
        raise TypeError("query must be a string.")

    if not query.strip():
        raise ValueError("query cannot be empty.")

    if not isinstance(store, VectorStore):
        raise TypeError("store must be a VectorStore instance.")

    if isinstance(k, bool) or not isinstance(k, int):
        raise TypeError("k must be an integer.")

    if k <= 0:
        raise ValueError("k must be greater than zero.")

    if model is None:
        model = create_embedding_model()

    query_vector = embed_query(query.strip(), model)
    return store.search(query_vector, k=k)
