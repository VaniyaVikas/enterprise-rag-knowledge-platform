
from src.bm25_retriever import BM25Retriever
from src.embeddings import create_embedding_model, embed_query
from src.vector_store import VectorStore


class HybridRetriever:
    """Combine BM25 and vector search using Reciprocal Rank Fusion."""

    def __init__(
        self,
        documents: list[str],
        vector_store: VectorStore,
        embedding_model=None,
        rrf_constant: int = 60,
    ) -> None:
        if not isinstance(documents, list):
            raise TypeError("documents must be a list.")

        if not documents:
            raise ValueError("documents cannot be empty.")

        if not all(
            isinstance(document, str) and document.strip()
            for document in documents
        ):
            raise ValueError("documents must contain non-empty strings.")

        if not isinstance(vector_store, VectorStore):
            raise TypeError("vector_store must be a VectorStore instance.")

        if documents != vector_store.documents:
            raise ValueError(
                "documents must match the vector store documents in order."
            )

        if isinstance(rrf_constant, bool) or not isinstance(rrf_constant, int):
            raise TypeError("rrf_constant must be an integer.")

        if rrf_constant <= 0:
            raise ValueError("rrf_constant must be greater than zero.")

        self.documents = documents.copy()
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.rrf_constant = rrf_constant
        self.bm25_retriever = BM25Retriever(self.documents)

    def search(self, query: str, k: int = 3) -> list[dict]:
        if not isinstance(query, str):
            raise TypeError("query must be a string.")

        if not query.strip():
            raise ValueError("query cannot be empty.")

        if isinstance(k, bool) or not isinstance(k, int):
            raise TypeError("k must be an integer.")

        if k <= 0:
            raise ValueError("k must be greater than zero.")

        bm25_results = self.bm25_retriever.search(
            query,
            k=len(self.documents),
        )

        model = self.embedding_model

        if model is None:
            model = create_embedding_model()
            self.embedding_model = model

        query_vector = embed_query(query, model)

        vector_results = self.vector_store.search(
            query_vector,
            k=len(self.documents),
        )

        fused_scores = {}
        bm25_ranks = {}
        vector_ranks = {}

        for rank, result in enumerate(bm25_results, start=1):
            index = result["index"]
            bm25_ranks[index] = rank
            fused_scores[index] = (
                1.0 / (self.rrf_constant + rank)
            )

        for rank, result in enumerate(vector_results, start=1):
            index = result["index"]
            vector_ranks[index] = rank
            fused_scores[index] = fused_scores.get(index, 0.0) + (
                1.0 / (self.rrf_constant + rank)
            )

        ranked_indices = sorted(
            fused_scores,
            key=lambda index: fused_scores[index],
            reverse=True,
        )

        results = []

        for index in ranked_indices[:k]:
            results.append(
                {
                    "document": self.documents[index],
                    "metadata": self.vector_store.metadata[index],
                    "score": fused_scores[index],
                    "index": index,
                    "bm25_rank": bm25_ranks.get(index),
                    "vector_rank": vector_ranks.get(index),
                }
            )

        return results
