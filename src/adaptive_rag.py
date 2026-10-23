
from collections.abc import Callable


class AdaptiveRAG:
    """Adapt retrieval using query routing and result-count fallback."""

    def __init__(
        self,
        router,
        min_results: int = 1,
        fallback_route: str | None = None,
    ):
        if not callable(getattr(router, "route", None)):
            raise TypeError("router must provide a callable route method")

        if isinstance(min_results, bool) or not isinstance(min_results, int):
            raise TypeError("min_results must be an integer")

        if min_results < 1:
            raise ValueError("min_results must be at least 1")

        retrievers = getattr(router, "retrievers", None)

        if not isinstance(retrievers, dict) or not retrievers:
            raise ValueError("router must provide registered retrievers")

        if fallback_route is not None:
            if fallback_route not in retrievers:
                raise ValueError("fallback_route must exist in retrievers")

        self.router = router
        self.min_results = min_results
        self.fallback_route = fallback_route

    def run(self, query: str, k: int = 3) -> dict:
        """Run routed retrieval and optionally fall back if results are sparse."""
        if not isinstance(query, str):
            raise TypeError("query must be a string")

        if not query.strip():
            raise ValueError("query must not be empty")

        if isinstance(k, bool) or not isinstance(k, int):
            raise TypeError("k must be an integer")

        if k < 1:
            raise ValueError("k must be at least 1")

        primary = self.router.route(query.strip(), k=k)
        results = primary["results"]
        selected_route = primary["route"]
        fallback_used = False

        if (
            len(results) < self.min_results
            and self.fallback_route is not None
            and self.fallback_route != selected_route
        ):
            fallback_retriever = self.router.retrievers[self.fallback_route]

            if callable(getattr(fallback_retriever, "search", None)):
                fallback_results = fallback_retriever.search(query.strip(), k=k)
            else:
                fallback_results = fallback_retriever(query.strip())

            results = fallback_results
            fallback_used = True

        return {
            "query": query.strip(),
            "route": selected_route,
            "fallback_route": (
                self.fallback_route if fallback_used else None
            ),
            "fallback_used": fallback_used,
            "results": results,
        }
