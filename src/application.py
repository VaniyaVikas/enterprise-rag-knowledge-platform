
"""Runnable FastAPI application for the Enterprise RAG Knowledge Platform."""

import os
from pathlib import Path

from dotenv import load_dotenv

from src.api import create_app
from src.document_processor import process_document_from_config
from src.embeddings import create_embedding_model, embed_texts
from src.groq_rag_pipeline import create_groq_rag_pipeline
from src.vector_store import VectorStore


ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")


def build_rag_pipeline():
    """Load documents, build the vector store, and configure Groq RAG."""
    raw_dir = ROOT_DIR / "data" / "raw"
    config_path = ROOT_DIR / "configs" / "chunking.yaml"

    document_paths = sorted(raw_dir.glob("*.txt"))

    if not document_paths:
        raise FileNotFoundError(
            f"No .txt documents found in {raw_dir}"
        )

    chunks = []

    for document_path in document_paths:
        chunks.extend(
            process_document_from_config(
                str(document_path),
                config_path=str(config_path),
            )
        )

    if not chunks:
        raise ValueError("Document processing produced no chunks.")

    model_name = os.getenv(
        "EMBEDDING_MODEL",
        "sentence-transformers/all-MiniLM-L6-v2",
    )

    embedding_model = create_embedding_model(model_name)

    documents = [chunk.text for chunk in chunks]
    metadata = [chunk.metadata for chunk in chunks]
    vectors = embed_texts(documents, embedding_model)

    store = VectorStore(dimension=len(vectors[0]))
    store.add(
        vectors=vectors,
        documents=documents,
        metadata=metadata,
    )

    retrieval_k = int(os.getenv("RETRIEVAL_K", "3"))

    return create_groq_rag_pipeline(
        store=store,
        embedding_model=embedding_model,
        k=retrieval_k,
    )


pipeline = build_rag_pipeline()
app = create_app(pipeline)
