
import pytest

from src.query_router import QueryRouter


class FakeRetriever:
    """Simple retriever for isolated routing tests."""

    def __init__(self, name):
        self.name = name
        self.last_query = None
        self.last_k = None

    def search(self, query, k=3):
        self.last_query = query
        self.last_k = k
        return [{"document": f"{self.name} result for: {query}"}]


def make_retrievers():
    return {
        "bm25": FakeRetriever("bm25"),
        "vector": FakeRetriever("vector"),
        "hybrid": FakeRetriever("hybrid"),
    }


def test_constructor_accepts_valid_retrievers():
    retrievers = make_retrievers()
    router = QueryRouter(retrievers)

    assert router.retrievers is retrievers
    assert router.default_route is None


def test_constructor_accepts_valid_default_route():
    router = QueryRouter(
        make_retrievers(),
        default_route="vector",
    )

    assert router.default_route == "vector"


@pytest.mark.parametrize("retrievers", [{}, [], None, "invalid"])
def test_constructor_rejects_invalid_retrievers(retrievers):
    with pytest.raises(
        ValueError,
        match="retrievers must be a non-empty dictionary",
    ):
        QueryRouter(retrievers)


@pytest.mark.parametrize("route", ["", "   ", 123])
def test_constructor_rejects_invalid_route_names(route):
    with pytest.raises(ValueError, match="route names must be non-empty"):
        QueryRouter({route: FakeRetriever("fake")})


def test_constructor_rejects_invalid_retriever():
    with pytest.raises(TypeError, match="must be callable"):
        QueryRouter({"vector": object()})


def test_constructor_rejects_unknown_default_route():
    with pytest.raises(ValueError, match="default_route must exist"):
        QueryRouter(make_retrievers(), default_route="unknown")


@pytest.mark.parametrize("query", ["", "   "])
def test_select_route_rejects_empty_query(query):
    router = QueryRouter(make_retrievers())

    with pytest.raises(ValueError, match="query must not be empty"):
        router.select_route(query)


def test_select_route_rejects_non_string_query():
    router = QueryRouter(make_retrievers())

    with pytest.raises(TypeError, match="query must be a string"):
        router.select_route(None)


@pytest.mark.parametrize(
    ("query", "expected_route"),
    [
        ("Find the exact error_code AUTH_401", "bm25"),
        ("Search for an identifier", "bm25"),
        ("Explain how RAG works", "vector"),
        ("How can an AI understand documents?", "vector"),
        ("Compare both approaches", "hybrid"),
        ("Find exact and semantic matches", "hybrid"),
    ],
)
def test_select_route_matches_keywords(query, expected_route):
    router = QueryRouter(make_retrievers())

    assert router.select_route(query) == expected_route


def test_select_route_is_case_insensitive():
    router = QueryRouter(make_retrievers())

    assert router.select_route("EXPLAIN RAG") == "vector"


def test_select_route_uses_default_when_no_keyword_matches():
    router = QueryRouter(
        make_retrievers(),
        default_route="bm25",
    )

    assert router.select_route("Tell me something") == "bm25"


def test_select_route_uses_only_registered_matching_route():
    router = QueryRouter(
        {"vector": FakeRetriever("vector")}
    )

    assert router.select_route("Explain RAG") == "vector"


def test_select_route_rejects_unknown_route_without_default():
    router = QueryRouter(make_retrievers())

    with pytest.raises(
        ValueError,
        match="could not determine route",
    ):
        router.select_route("Tell me something")


def test_route_executes_selected_retriever():
    retrievers = make_retrievers()
    router = QueryRouter(retrievers)

    result = router.route("Explain RAG", k=2)

    assert result["route"] == "vector"
    assert result["query"] == "Explain RAG"
    assert retrievers["vector"].last_query == "Explain RAG"
    assert retrievers["vector"].last_k == 2
    assert "vector result" in result["results"][0]["document"]


def test_route_returns_bm25_results_for_exact_query():
    retrievers = make_retrievers()
    router = QueryRouter(retrievers)

    result = router.route("Find the exact error_code AUTH_401")

    assert result["route"] == "bm25"
    assert retrievers["bm25"].last_query == (
        "Find the exact error_code AUTH_401"
    )


def test_route_returns_hybrid_results_for_combined_query():
    retrievers = make_retrievers()
    router = QueryRouter(retrievers)

    result = router.route("Compare both approaches")

    assert result["route"] == "hybrid"
    assert "hybrid result" in result["results"][0]["document"]


def test_route_supports_direct_callable_retriever():
    received_queries = []

    def retriever(query):
        received_queries.append(query)
        return [{"document": "Callable retriever result"}]

    router = QueryRouter({"custom": retriever})

    result = router.route("A general query")

    assert result["route"] == "custom"
    assert received_queries == ["A general query"]
    assert result["results"][0]["document"] == (
        "Callable retriever result"
    )
