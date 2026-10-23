
"""Factory for creating a production RAG pipeline backed by Groq."""

from src.groq_answer_generator import create_groq_answer_generator
from src.production_rag_pipeline import ProductionRAGPipeline
from src.vector_store import VectorStore


def create_groq_rag_pipeline(
    store: VectorStore,
    embedding_model=None,
    k: int = 3,
    groq_model: str | None = None,
    temperature: float = 0.0,
) -> ProductionRAGPipeline:
    """Create a production RAG pipeline using Groq for answer generation."""
    if not isinstance(store, VectorStore):
        raise TypeError("store must be a VectorStore instance.")

    answer_generator = create_groq_answer_generator(
        model=groq_model,
        temperature=temperature,
    )

    return ProductionRAGPipeline(
        store=store,
        answer_generator=answer_generator,
        model=embedding_model,
        k=k,
    )
