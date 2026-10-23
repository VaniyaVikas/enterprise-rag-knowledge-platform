
from collections.abc import Callable


class RAGFusionRetriever:
    """Fuse multiple query rankings using Reciprocal Rank Fusion."""

    def __init__(
        self,
        retriever,
        query_generator: Callable[[str], list[str]],
        rrf_constant: int = 60,
    ) -> None:
        if not callable(query_generator):
            raise TypeError("query_generator must be callable.")

        if not callable(retriever) and not callable(
            getattr(retriever, "search", None)
        ):
            raise TypeError(
                "retriever must be callable or provide a callable search method."
            )

        if isinstance(rrf_constant, bool) or not isinstance(
            rrf_constant, int
        ):
            raise TypeError("rrf_constant must be an integer.")

        if rrf_constant <= 0:
            raise ValueError("rrf_constant must be greater than zero.")

        self.retriever = retriever
        self.query_generator = query_generator
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

    def generate_queries(self, query: str) -> list[str]:
        query = self._validate_query(query)
        generated = self.query_generator(query)

        if not isinstance(generated, list):
            raise TypeError("query_generator must return a list.")

        queries = []
        seen = set()

        for item in [query, *generated]:
            if not isinstance(item, str):
                raise TypeError("generated queries must be strings.")

            if not item.strip():
                raise ValueError("generated queries cannot be empty.")

            normalized = item.strip()
            key = normalized.casefold()

            if key not in seen:
                seen.add(key)
                queries.append(normalized)

        return queries

    def _search(self, query: str, k: int) -> list[dict]:
        search_method = getattr(self.retriever, "search", None)

        if callable(search_method):
            return search_method(query, k=k)

        return self.retriever(query)

    def search(self, query: str, k: int = 3) -> list[dict]:
        query = self._validate_query(query)
        self._validate_k(k)

        queries = self.generate_queries(query)
        fused = {}

        for generated_query in queries:
            results = self._search(generated_query, k)

            if not isinstance(results, list):
                raise TypeError("retriever must return a list of results.")

            seen_in_query = set()

            for rank, result in enumerate(results, start=1):
                if not isinstance(result, dict):
                    raise TypeError("each retrieval result must be a dictionary.")

                if "index" not in result:
                    raise ValueError(
                        "each retrieval result must contain an 'index' field."
                    )

                index = result["index"]

                if index in seen_in_query:
                    continue

                seen_in_query.add(index)

                if index not in fused:
                    fused[index] = result.copy()
                    fused[index]["rrf_score"] = 0.0
                    fused[index]["matched_queries"] = []

                fused[index]["rrf_score"] += (
                    1.0 / (self.rrf_constant + rank)
                )
                fused[index]["matched_queries"].append(generated_query)

        ranked_results = sorted(
            fused.values(),
            key=lambda item: item["rrf_score"],
            reverse=True,
        )

        return ranked_results[:k]
