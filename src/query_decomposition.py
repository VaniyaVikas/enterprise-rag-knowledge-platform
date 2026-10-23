
from collections.abc import Callable


class QueryDecomposer:
    """Break complex queries into sub-queries and retrieve their results."""

    def __init__(
        self,
        decomposer: Callable[[str], list[str]],
        retriever: Callable[[str], list[dict]] | object,
    ):
        if not callable(decomposer):
            raise TypeError("decomposer must be callable")

        if not callable(retriever) and not callable(
            getattr(retriever, "search", None)
        ):
            raise TypeError(
                "retriever must be callable or provide a callable search method"
            )

        self.decomposer = decomposer
        self.retriever = retriever

    def decompose(self, query: str) -> list[str]:
        """Generate and validate sub-queries."""
        if not isinstance(query, str):
            raise TypeError("query must be a string")

        if not query.strip():
            raise ValueError("query must not be empty")

        sub_queries = self.decomposer(query.strip())

        if not isinstance(sub_queries, list):
            raise TypeError("decomposer must return a list")

        if not sub_queries:
            raise ValueError("decomposer must return at least one sub-query")

        validated_queries = []
        seen_queries = set()

        for sub_query in sub_queries:
            if not isinstance(sub_query, str):
                raise TypeError("each sub-query must be a string")

            normalized_query = sub_query.strip()

            if not normalized_query:
                raise ValueError("sub-queries must not be empty")

            deduplication_key = normalized_query.casefold()

            if deduplication_key not in seen_queries:
                seen_queries.add(deduplication_key)
                validated_queries.append(normalized_query)

        return validated_queries

    def retrieve(self, query: str) -> dict:
        """Retrieve results for every sub-query."""
        sub_queries = self.decompose(query)
        sub_results = []

        for sub_query in sub_queries:
            if callable(getattr(self.retriever, "search", None)):
                results = self.retriever.search(sub_query)
            else:
                results = self.retriever(sub_query)

            sub_results.append(
                {
                    "sub_query": sub_query,
                    "results": results,
                }
            )

        return {
            "original_query": query.strip(),
            "sub_queries": sub_queries,
            "sub_results": sub_results,
        }
