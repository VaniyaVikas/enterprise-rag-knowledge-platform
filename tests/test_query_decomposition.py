
import pytest

from src.query_decomposition import QueryDecomposer


class FakeRetriever:
    """Fake retriever that records queries and returns predictable results."""

    def __init__(self):
        self.queries = []

    def search(self, query):
        self.queries.append(query)
        return [{"document": f"Result for: {query}"}]


def make_decomposer(sub_queries=None, retriever=None):
    if sub_queries is None:
        sub_queries = ["Python basics", "Python applications"]

    if retriever is None:
        retriever = FakeRetriever()

    return QueryDecomposer(
        decomposer=lambda query: sub_queries,
        retriever=retriever,
    )


def test_constructor_accepts_valid_dependencies():
    retriever = FakeRetriever()
    decomposer = lambda query: ["sub-query"]

    instance = QueryDecomposer(decomposer, retriever)

    assert instance.decomposer is decomposer
    assert instance.retriever is retriever


def test_constructor_rejects_non_callable_decomposer():
    with pytest.raises(TypeError, match="decomposer must be callable"):
        QueryDecomposer(None, FakeRetriever())


@pytest.mark.parametrize("retriever", [None, object(), "invalid"])
def test_constructor_rejects_invalid_retriever(retriever):
    with pytest.raises(
        TypeError,
        match="retriever must be callable or provide a callable search method",
    ):
        QueryDecomposer(lambda query: ["sub-query"], retriever)


@pytest.mark.parametrize("query", ["", "   ", "\n\t"])
def test_decompose_rejects_empty_query(query):
    instance = make_decomposer()

    with pytest.raises(ValueError, match="query must not be empty"):
        instance.decompose(query)


def test_decompose_rejects_non_string_query():
    instance = make_decomposer()

    with pytest.raises(TypeError, match="query must be a string"):
        instance.decompose(None)


def test_decompose_strips_query_before_calling_decomposer():
    received_queries = []
    instance = QueryDecomposer(
        decomposer=lambda query: (
            received_queries.append(query) or ["sub-query"]
        ),
        retriever=FakeRetriever(),
    )

    instance.decompose("  Explain Python  ")

    assert received_queries == ["Explain Python"]


def test_decompose_returns_sub_queries():
    instance = make_decomposer(
        ["Python basics", "Python applications"]
    )

    assert instance.decompose("Explain Python") == [
        "Python basics",
        "Python applications",
    ]


def test_decompose_rejects_non_list_result():
    instance = QueryDecomposer(
        decomposer=lambda query: "not a list",
        retriever=FakeRetriever(),
    )

    with pytest.raises(TypeError, match="decomposer must return a list"):
        instance.decompose("Explain Python")


def test_decompose_rejects_empty_result():
    instance = make_decomposer([])

    with pytest.raises(
        ValueError,
        match="decomposer must return at least one sub-query",
    ):
        instance.decompose("Explain Python")


def test_decompose_rejects_non_string_sub_query():
    instance = make_decomposer(["Python", 123])

    with pytest.raises(TypeError, match="each sub-query must be a string"):
        instance.decompose("Explain Python")


def test_decompose_rejects_empty_sub_query():
    instance = make_decomposer(["Python", "   "])

    with pytest.raises(
        ValueError,
        match="sub-queries must not be empty",
    ):
        instance.decompose("Explain Python")


def test_decompose_strips_sub_queries():
    instance = make_decomposer(["  Python basics  ", " Python usage "])

    assert instance.decompose("Explain Python") == [
        "Python basics",
        "Python usage",
    ]


def test_decompose_removes_case_insensitive_duplicates():
    instance = make_decomposer(
        ["Python basics", "python basics", "PYTHON BASICS", "Python usage"]
    )

    assert instance.decompose("Explain Python") == [
        "Python basics",
        "Python usage",
    ]


def test_decompose_preserves_first_duplicate_spelling():
    instance = make_decomposer(["  Python Basics ", "python basics"])

    assert instance.decompose("Explain Python") == ["Python Basics"]


def test_retrieve_searches_every_sub_query():
    retriever = FakeRetriever()
    instance = make_decomposer(
        ["Python basics", "Python applications"],
        retriever,
    )

    instance.retrieve("Explain Python")

    assert retriever.queries == [
        "Python basics",
        "Python applications",
    ]


def test_retrieve_returns_expected_structure():
    instance = make_decomposer(
        ["Python basics", "Python applications"]
    )

    result = instance.retrieve("Explain Python")

    assert result == {
        "original_query": "Explain Python",
        "sub_queries": ["Python basics", "Python applications"],
        "sub_results": [
            {
                "sub_query": "Python basics",
                "results": [{"document": "Result for: Python basics"}],
            },
            {
                "sub_query": "Python applications",
                "results": [
                    {"document": "Result for: Python applications"}
                ],
            },
        ],
    }


def test_retrieve_strips_original_query_in_result():
    instance = make_decomposer()

    result = instance.retrieve("  Explain Python  ")

    assert result["original_query"] == "Explain Python"


def test_retrieve_supports_direct_callable_retriever():
    received_queries = []

    def retriever(query):
        received_queries.append(query)
        return [{"document": f"Callable result for: {query}"}]

    instance = make_decomposer(
        ["Python basics", "Python applications"],
        retriever,
    )

    result = instance.retrieve("Explain Python")

    assert received_queries == ["Python basics", "Python applications"]
    assert len(result["sub_results"]) == 2
    assert result["sub_results"][0]["results"] == [
        {"document": "Callable result for: Python basics"}
    ]
