import numpy as np

from src.embeddings import create_embedding_model, embed_query
from src.vector_store import VectorStore


class MMRRetriever:
    """Retrieve relevant and diverse documents using MMR."""

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_model=None,
        lambda_mult: float = 0.5,
    ) -> None:
        if not isinstance(vector_store, VectorStore):
            raise TypeError(
                "vector_store must be a VectorStore instance."
            )

        if not vector_store.documents:
            raise ValueError(
                "vector_store must contain documents."
            )

        if (
            isinstance(lambda_mult, bool)
            or not isinstance(lambda_mult, (int, float))
        ):
            raise TypeError("lambda_mult must be a number.")

        if not 0.0 <= lambda_mult <= 1.0:
            raise ValueError(
                "lambda_mult must be between 0 and 1."
            )

        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.lambda_mult = float(lambda_mult)

    @staticmethod
    def _cosine_similarity(
        first: list[float],
        second: list[float],
    ) -> float:
        first_vector = np.asarray(first, dtype=np.float32)
        second_vector = np.asarray(second, dtype=np.float32)

        first_norm = np.linalg.norm(first_vector)
        second_norm = np.linalg.norm(second_vector)

        if first_norm == 0 or second_norm == 0:
            return 0.0

        return float(
            np.dot(first_vector, second_vector)
            / (first_norm * second_norm)
        )

    def search(
        self,
        query: str,
        k: int = 3,
        fetch_k: int = 10,
    ) -> list[dict]:
        if not isinstance(query, str):
            raise TypeError("query must be a string.")

        if not query.strip():
            raise ValueError("query cannot be empty.")

        if isinstance(k, bool) or not isinstance(k, int):
            raise TypeError("k must be an integer.")

        if k <= 0:
            raise ValueError("k must be greater than zero.")

        if isinstance(fetch_k, bool) or not isinstance(fetch_k, int):
            raise TypeError("fetch_k must be an integer.")

        if fetch_k <= 0:
            raise ValueError("fetch_k must be greater than zero.")

        if fetch_k < k:
            raise ValueError("fetch_k must be greater than or equal to k.")

        model = self.embedding_model

        if model is None:
            model = create_embedding_model()
            self.embedding_model = model

        query_vector = embed_query(query, model)

        candidate_results = self.vector_store.search(
            query_vector=query_vector,
            k=min(fetch_k, len(self.vector_store.documents)),
        )

        if not candidate_results:
            return []

        candidates = []

        for result in candidate_results:
            document_vector = self.vector_store.index.reconstruct(
                result["index"]
            )

            candidates.append(
                {
                    **result,
                    "vector": document_vector.tolist(),
                }
            )

        selected = []
        selected_indices = set()
        selection_limit = min(k, len(candidates))

        while len(selected) < selection_limit:
            best_candidate = None
            best_score = float("-inf")

            for candidate in candidates:
                index = candidate["index"]

                if index in selected_indices:
                    continue

                relevance = self._cosine_similarity(
                    query_vector,
                    candidate["vector"],
                )

                if selected:
                    redundancy = max(
                        self._cosine_similarity(
                            candidate["vector"],
                            item["vector"],
                        )
                        for item in selected
                    )
                else:
                    redundancy = 0.0

                mmr_score = (
                    self.lambda_mult * relevance
                    - (1.0 - self.lambda_mult) * redundancy
                )

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_candidate = candidate

            if best_candidate is None:
                break

            chosen = best_candidate.copy()
            chosen["score"] = float(best_score)
            chosen.pop("vector")

            selected.append(
                {
                    **chosen,
                    "vector": best_candidate["vector"],
                }
            )
            selected_indices.add(best_candidate["index"])

        return [
            {
                key: value
                for key, value in item.items()
                if key != "vector"
            }
            for item in selected
        ]
