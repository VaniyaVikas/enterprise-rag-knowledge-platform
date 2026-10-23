
"""Embedding model creation and text/query embedding utilities."""

from langchain_huggingface import HuggingFaceEmbeddings


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def create_embedding_model(
    model_name: str = DEFAULT_EMBEDDING_MODEL,
) -> HuggingFaceEmbeddings:
    """Create a CPU-based Hugging Face embedding model."""
    if not isinstance(model_name, str):
        raise TypeError("model_name must be a string.")

    if not model_name.strip():
        raise ValueError("model_name cannot be empty.")

    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def embed_texts(
    texts: list[str],
    model: HuggingFaceEmbeddings,
) -> list[list[float]]:
    """Convert a list of documents into embedding vectors."""
    if not isinstance(texts, list):
        raise TypeError("texts must be a list of strings.")

    if not all(isinstance(text, str) for text in texts):
        raise TypeError("all texts must be strings.")

    if not isinstance(model, HuggingFaceEmbeddings):
        raise TypeError("model must be a HuggingFaceEmbeddings instance.")

    if not texts:
        return []

    if any(not text.strip() for text in texts):
        raise ValueError("texts cannot contain empty strings.")

    return model.embed_documents(texts)


def embed_query(
    query: str,
    model: HuggingFaceEmbeddings,
) -> list[float]:
    """Convert a search query into an embedding vector."""
    if not isinstance(query, str):
        raise TypeError("query must be a string.")

    if not query.strip():
        raise ValueError("query cannot be empty.")

    if not isinstance(model, HuggingFaceEmbeddings):
        raise TypeError("model must be a HuggingFaceEmbeddings instance.")

    return model.embed_query(query.strip())
