
import pytest

from src.adaptive_rag import AdaptiveRAG
from src.query_router import QueryRouter


class FakeRetriever:
    """Fake retriever with configurable results."""

    def __init__(self, name, results=None):
        self.name = name
        self.results = results if results is not None else []
        self.queries = []
        self.k_values = []

    def search(self, query, k=3):
        self.queries.append(query)
        self.k_values.append(k)
        return self.results


def make_router(primary_results=None, fallback_results=None):
    retrievers = {
        "vector": FakeRetriever("vector", primary_results),
        "bm25": FakeRetriever("bm25", fallback_results),
    }
    return QueryRouter(
        retrievers,
        default_route="vector",
    )


def test_constructor_accepts_valid_configuration():
    router = make_router([{"document": "A"}])
    adaptive_rag = AdaptiveRAG(
        router,
        min_results=2,
        fallback_route="bm25",
    )

    assert adaptive_rag.router is router
    assert adaptive_rag.min_results == 2
    assert adaptive_rag.fallback_route == "bm25"


def test_constructor_accepts_no_fallback_route():
    router = make_router()

    adaptive_rag = AdaptiveRAG(router)

    assert adaptive_rag.fallback_route is None
    assert adaptive_rag.min_results == 1


@pytest.mark.parametrize("router", [None, object(), "invalid"])
def test_constructor_rejects_invalid_router(router):
    with pytest.raises(
        TypeError,
        match="router must provide a callable route method",
    ):
        AdaptiveRAG(router)


@pytest.mark.parametrize("min_results", [True, False, 1.5, "2", None])
def test_constructor_rejects_non_integer_min_results(min_results):
    with pytest.raises(TypeError, match="min_results must be an integer"):
        AdaptiveRAG(make_router(), min_results=min_results)


@pytest.mark.parametrize("min_results", [0, -1, -5])
def test_constructor_rejects_non_positive_min_results(min_results):
    with pytest.raises(ValueError, match="min_results must be at least 1"):
        AdaptiveRAG(make_router(), min_results=min_results)


def test_constructor_rejects_unknown_fallback_route():
    with pytest.raises(
        ValueError,
        match="fallback_route must exist in retrievers",
    ):
        AdaptiveRAG(make_router(), fallback_route="unknown")


@pytest.mark.parametrize("query", ["", "   ", "\n\t"])
def test_run_rejects_empty_query(query):
    adaptive_rag = AdaptiveRAG(make_router())

    with pytest.raises(ValueError, match="query must not be empty"):
        adaptive_rag.run(query)


def test_run_rejects_non_string_query():
    adaptive_rag = AdaptiveRAG(make_router())

    with pytest.raises(TypeError, match="query must be a string"):
        adaptive_rag.run(None)


@pytest.mark.parametrize("k", [True, False, 1.5, "3", None])
def test_run_rejects_non_integer_k(k):
    adaptive_rag = AdaptiveRAG(make_router())

    with pytest.raises(TypeError, match="k must be an integer"):
        adaptive_rag.run("Explain RAG", k=k)


@pytest.mark.parametrize("k", [0, -1, -3])
def test_run_rejects_non_positive_k(k):
    adaptive_rag = AdaptiveRAG(make_router())

    with pytest.raises(ValueError, match="k must be at least 1"):
        adaptive_rag.run("Explain RAG", k=k)


def test_run_returns_primary_results_when_threshold_is_met():
    primary_results = [
        {"document": "A"},
        {"document": "B"},
    ]
    router = make_router(primary_results, [{"document": "Fallback"}])
    adaptive_rag = AdaptiveRAG(
        router,
        min_results=2,
        fallback_route="bm25",
    )

    result = adaptive_rag.run("Explain RAG", k=2)

    assert result["route"] == "vector"
    assert result["results"] == primary_results
    assert result["fallback_used"] is False
    assert result["fallback_route"] is None
    assert router.retrievers["vector"].queries == ["Explain RAG"]
    assert router.retrievers["vector"].k_values == [2]
    assert router.retrievers["bm25"].queries == []


def test_run_uses_fallback_when_primary_results_are_insufficient():
    fallback_results = [{"document": "Fallback result"}]
    router = make_router([], fallback_results)
    adaptive_rag = AdaptiveRAG(
        router,
        min_results=1,
        fallback_route="bm25",
    )

    result = adaptive_rag.run("Explain RAG", k=4)

    assert result["fallback_used"] is True
    assert result["fallback_route"] == "bm25"
    assert result["results"] == fallback_results
    assert router.retrievers["bm25"].queries == ["Explain RAG"]
    assert router.retrievers["bm25"].k_values == [4]


def test_run_does_not_fallback_when_no_fallback_is_configured():
    router = make_router([])
    adaptive_rag = AdaptiveRAG(router, min_results=1)

    result = adaptive_rag.run("Explain RAG")

    assert result["results"] == []
    assert result["fallback_used"] is False
    assert result["fallback_route"] is None


def test_run_does_not_fallback_when_primary_route_is_fallback_route():
    router = make_router([])
    adaptive_rag = AdaptiveRAG(
        router,
        fallback_route="vector",
    )

    result = adaptive_rag.run("Explain RAG")

    assert result["fallback_used"] is False
    assert router.retrievers["vector"].queries == ["Explain RAG"]


def test_run_strips_query_before_retrieval():
    router = make_router([{"document": "A"}])
    adaptive_rag = AdaptiveRAG(router)

    result = adaptive_rag.run("  Explain RAG  ")

    assert result["query"] == "Explain RAG"
    assert router.retrievers["vector"].queries == ["Explain RAG"]


def test_run_returns_expected_response_structure():
    primary_results = [{"document": "Relevant result"}]
    router = make_router(primary_results)
    adaptive_rag = AdaptiveRAG(router)

    result = adaptive_rag.run("Explain RAG")

    assert result == {
        "query": "Explain RAG",
        "route": "vector",
        "fallback_route": None,
        "fallback_used": False,
        "results": primary_results,
    }


def test_run_supports_direct_callable_fallback_retriever():
    primary = FakeRetriever("vector", [])
    fallback_calls = []

    def fallback(query):
        fallback_calls.append(query)
        return [{"document": "Callable fallback result"}]

    router = QueryRouter(
        {
            "vector": primary,
            "custom": fallback,
        },
        default_route="vector",
    )
    adaptive_rag = AdaptiveRAG(
        router,
        fallback_route="custom",
    )

    result = adaptive_rag.run("Explain RAG")

    assert fallback_calls == ["Explain RAG"]
    assert result["fallback_used"] is True
    assert result["results"] == [
        {"document": "Callable fallback result"}
    ]

