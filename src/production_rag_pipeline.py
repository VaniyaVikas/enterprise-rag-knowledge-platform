
"""Production-ready RAG pipeline orchestration."""

from typing import Any, Callable

from src.answer_pipeline import AnswerPipeline
from src.retrieval import retrieve_documents
from src.vector_store import VectorStore


class ProductionRAGPipeline:
    """Retrieve relevant documents and produce a validated RAG response."""

    def __init__(
        self,
        store: VectorStore,
        answer_generator: Callable[[str, str], str] | None = None,
        model: Any = None,
        k: int = 3,
        answer_pipeline: AnswerPipeline | None = None,
    ) -> None:
        if not isinstance(store, VectorStore):
            raise TypeError("store must be a VectorStore instance.")

        if isinstance(k, bool) or not isinstance(k, int):
            raise TypeError("k must be an integer.")

        if k <= 0:
            raise ValueError("k must be greater than zero.")

        if answer_pipeline is None:
            if not callable(answer_generator):
                raise TypeError(
                    "answer_generator must be callable when "
                    "answer_pipeline is not provided."
                )

            answer_pipeline = AnswerPipeline(
                answer_generator=answer_generator
            )
        elif not callable(getattr(answer_pipeline, "run", None)):
            raise TypeError(
                "answer_pipeline must provide a callable run method."
            )

        self.store = store
        self.model = model
        self.k = k
        self.answer_pipeline = answer_pipeline

    @staticmethod
    def _validate_query(query: str) -> str:
        """Validate and normalize the user's query."""
        if not isinstance(query, str):
            raise TypeError("query must be a string.")

        if not query.strip():
            raise ValueError("query cannot be empty.")

        return query.strip()

    def run(self, query: str) -> dict:
        """Retrieve documents and return a validated RAG response."""
        query = self._validate_query(query)

        documents = retrieve_documents(
            query=query,
            store=self.store,
            model=self.model,
            k=self.k,
        )

        if not isinstance(documents, list):
            raise TypeError("retrieval must return a list of documents.")

        result = self.answer_pipeline.run(
            query=query,
            retrieved_documents=documents,
        )

        if not isinstance(result, dict):
            raise TypeError(
                "answer_pipeline.run must return a dictionary."
            )

        response = result.copy()
        response["retrieval"] = {
            "requested_k": self.k,
            "document_count": len(documents),
        }

        return response
