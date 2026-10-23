
from collections.abc import Callable


class EnsembleRetriever:
    """Combine multiple retrievers using weighted reciprocal rank fusion."""

    def __init__(
        self,
        retrievers: dict[str, object],
        weights: dict[str, float] | None = None,
        rrf_constant: int = 60,
    ) -> None:
        if not isinstance(retrievers, dict):
            raise TypeError("retrievers must be a dictionary.")

        if not retrievers:
            raise ValueError("retrievers cannot be empty.")

        for name, retriever in retrievers.items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError(
                    "retriever names must be non-empty strings."
                )

            if not callable(retriever) and not callable(
                getattr(retriever, "search", None)
            ):
                raise TypeError(
                    "each retriever must be callable or provide a callable search method."
                )

        if isinstance(rrf_constant, bool) or not isinstance(
            rrf_constant, int
        ):
            raise TypeError("rrf_constant must be an integer.")

        if rrf_constant <= 0:
            raise ValueError("rrf_constant must be greater than zero.")

        if weights is None:
            weights = {
                name: 1.0 / len(retrievers)
                for name in retrievers
            }

        if not isinstance(weights, dict):
            raise TypeError("weights must be a dictionary.")

        if set(weights) != set(retrievers):
            raise ValueError(
                "weights must contain exactly the configured retriever names."
            )

        for name, weight in weights.items():
            if (
                isinstance(weight, bool)
                or not isinstance(weight, (int, float))
            ):
                raise TypeError("each weight must be a number.")

            if weight < 0:
                raise ValueError("weights cannot be negative.")

        if sum(weights.values()) <= 0:
            raise ValueError("at least one weight must be greater than zero.")

        self.retrievers = retrievers.copy()
        self.weights = {
            name: float(weight)
            for name, weight in weights.items()
        }
        self.rrf_constant = rrf_constant

    @staticmethod
    def _validate_query(query: str) -> str:
        if not isinstance(query, str):
            raise TypeError("query must be a string.")

        if not query.strip():
            raise ValueError("query cannot be empty.")

        return query.strip()

    @staticmethod
    def _validate_k(k: int) -> None:
        if isinstance(k, bool) or not isinstance(k, int):
            raise TypeError("k must be an integer.")

        if k <= 0:
            raise ValueError("k must be greater than zero.")

    def _search_retriever(
        self,
        retriever: object,
        query: str,
        k: int,
    ) -> list[dict]:
        search_method = getattr(retriever, "search", None)

        if callable(search_method):
            results = search_method(query, k=k)
        elif isinstance(retriever, Callable):
            results = retriever(query)
        else:
            raise TypeError("retriever is not searchable.")

        if not isinstance(results, list):
            raise TypeError("each retriever must return a list of results.")

        return results

    def search(self, query: str, k: int = 3) -> list[dict]:
        query = self._validate_query(query)
        self._validate_k(k)

        fused_results = {}

        for name, retriever in self.retrievers.items():
            results = self._search_retriever(retriever, query, k)
            weight = self.weights[name]
            seen_indices = set()

            for rank, result in enumerate(results, start=1):
                if not isinstance(result, dict):
                    raise TypeError(
                        "each retrieval result must be a dictionary."
                    )

                if "index" not in result:
                    raise ValueError(
                        "each retrieval result must contain an 'index' field."
                    )

                index = result["index"]

                try:
                    hash(index)
                except TypeError as exc:
                    raise TypeError(
                        "retrieval result index must be hashable."
                    ) from exc

                if index in seen_indices:
                    continue

                seen_indices.add(index)

                if index not in fused_results:
                    fused_results[index] = result.copy()
                    fused_results[index]["ensemble_score"] = 0.0
                    fused_results[index]["matched_retrievers"] = []

                fused_results[index]["ensemble_score"] += (
                    weight / (self.rrf_constant + rank)
                )
                fused_results[index]["matched_retrievers"].append(name)

        ranked_results = sorted(
            fused_results.values(),
            key=lambda item: item["ensemble_score"],
            reverse=True,
        )

        return ranked_results[:k]
