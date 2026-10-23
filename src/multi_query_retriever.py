
from src.embeddings import create_embedding_model
from src.retrieval import retrieve_documents
from src.vector_store import VectorStore


class MultiQueryRetriever:
    """Retrieve documents using multiple query variations."""

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_model=None,
    ) -> None:
        if not isinstance(vector_store, VectorStore):
            raise TypeError(
                "vector_store must be a VectorStore instance."
            )

        if not vector_store.documents:
            raise ValueError(
                "vector_store must contain documents."
            )

        self.vector_store = vector_store
        self.embedding_model = embedding_model

    @staticmethod
    def generate_queries(query: str) -> list[str]:
        if not isinstance(query, str):
            raise TypeError("query must be a string.")

        if not query.strip():
            raise ValueError("query cannot be empty.")

        original = query.strip()

        variations = [
            original,
            f"What is {original}",
            f"How does {original} work",
            f"Explain {original}",
        ]

        unique_queries = []
        seen = set()

        for item in variations:
            normalized = item.casefold()

            if normalized not in seen:
                seen.add(normalized)
                unique_queries.append(item)

        return unique_queries

    def search(
        self,
        query: str,
        k: int = 3,
    ) -> list[dict]:
        if not isinstance(query, str):
            raise TypeError("query must be a string.")

        if not query.strip():
            raise ValueError("query cannot be empty.")

        if isinstance(k, bool) or not isinstance(k, int):
            raise TypeError("k must be an integer.")

        if k <= 0:
            raise ValueError("k must be greater than zero.")

        model = self.embedding_model

        if model is None:
            model = create_embedding_model()
            self.embedding_model = model

        queries = self.generate_queries(query)
        merged_results = {}
        
        for generated_query in queries:
            results = retrieve_documents(
                query=generated_query,
                store=self.vector_store,
                model=model,
                k=k,
            )

            for result in results:
                index = result["index"]

                if index not in merged_results:
                    merged_results[index] = result.copy()
                    merged_results[index]["matched_queries"] = [
                        generated_query
                    ]
                else:
                    merged_results[index]["matched_queries"].append(
                        generated_query
                    )

        ranked_results = sorted(
            merged_results.values(),
            key=lambda item: len(item["matched_queries"]),
            reverse=True,
        )

        return ranked_results[:k]
