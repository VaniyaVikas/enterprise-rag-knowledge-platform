
from collections.abc import Callable


class QueryRouter:
    """Route queries to the most appropriate registered retriever."""

    def __init__(
        self,
        retrievers: dict[str, Callable],
        default_route: str | None = None,
    ):
        if not isinstance(retrievers, dict) or not retrievers:
            raise ValueError("retrievers must be a non-empty dictionary")

        for route, retriever in retrievers.items():
            if not isinstance(route, str) or not route.strip():
                raise ValueError("route names must be non-empty strings")

            if not callable(retriever) and not callable(
                getattr(retriever, "search", None)
            ):
                raise TypeError(
                    f"retriever for route '{route}' must be callable "
                    "or provide a callable search method"
                )

        if default_route is not None and default_route not in retrievers:
            raise ValueError("default_route must exist in retrievers")

        self.retrievers = retrievers
        self.default_route = default_route

    def select_route(self, query: str) -> str:
        """Select a route using deterministic keyword-based rules."""
        if not isinstance(query, str):
            raise TypeError("query must be a string")

        if not query.strip():
            raise ValueError("query must not be empty")

        normalized_query = query.casefold()

        keyword_routes = (
            (
                "hybrid",
                (
                    " and ",
                    "both",
                    "compare",
                    "exact and semantic",
                    "keyword and meaning",
                ),
            ),
            (
                "bm25",
                (
                    "exact",
                    "error_code",
                    "error code",
                    "identifier",
                    "keyword",
                ),
            ),
            (
                "vector",
                (
                    "explain",
                    "how does",
                    "how can",
                    "meaning",
                    "understand",
                    "concept",
                ),
            ),
        )

        for route, keywords in keyword_routes:
            if route in self.retrievers and any(
                keyword in normalized_query for keyword in keywords
            ):
                return route

        if self.default_route is not None:
            return self.default_route

        if len(self.retrievers) == 1:
            return next(iter(self.retrievers))

        raise ValueError(
            "could not determine route and no default_route is configured"
        )

    def route(self, query: str, k: int = 3) -> dict:
        """Select a retriever, execute the search, and return its results."""
        selected_route = self.select_route(query)
        retriever = self.retrievers[selected_route]

        if callable(getattr(retriever, "search", None)):
            results = retriever.search(query, k=k)
        else:
            results = retriever(query)

        return {
            "route": selected_route,
            "query": query,
            "results": results,
        }
