
import pytest

from src.rag_fusion import RAGFusionRetriever


class FakeRetriever:
    """Return predefined results for each query."""

    def __init__(self, results_by_query):
        self.results_by_query = results_by_query
        self.calls = []

    def search(self, query, k=3):
        self.calls.append((query, k))
        return self.results_by_query.get(query, [])[:k]


def make_result(index, document=None):
    return {
        "index": index,
        "document": document or f"Document {index}",
        "metadata": {},
        "score": 1.0,
    }


def make_fusion(results_by_query, generated_queries=None, rrf_constant=60):
    retriever = FakeRetriever(results_by_query)

    if generated_queries is None:
        generated_queries = ["alternate query"]

    fusion = RAGFusionRetriever(
        retriever=retriever,
        query_generator=lambda query: generated_queries,
        rrf_constant=rrf_constant,
    )

    return fusion, retriever


def test_generate_queries_includes_original_and_alternatives():
    fusion, _ = make_fusion({}, ["alternative one", "alternative two"])

    assert fusion.generate_queries("original query") == [
        "original query",
        "alternative one",
        "alternative two",
    ]


def test_generate_queries_removes_case_insensitive_duplicates():
    fusion, _ = make_fusion(
        {},
        ["ORIGINAL QUERY", "another query", "Another Query"],
    )

    assert fusion.generate_queries("original query") == [
        "original query",
        "another query",
    ]


def test_generate_queries_strips_whitespace():
    fusion, _ = make_fusion({}, ["  alternative query  "])

    assert fusion.generate_queries("  original query  ") == [
        "original query",
        "alternative query",
    ]


@pytest.mark.parametrize("query", [None, 123, []])
def test_generate_queries_rejects_non_string_query(query):
    fusion, _ = make_fusion({})

    with pytest.raises(TypeError, match="query must be a string"):
        fusion.generate_queries(query)


@pytest.mark.parametrize("query", ["", "   ", "\n"])
def test_generate_queries_rejects_empty_query(query):
    fusion, _ = make_fusion({})

    with pytest.raises(ValueError, match="query cannot be empty"):
        fusion.generate_queries(query)


def test_generate_queries_rejects_non_list_generator_result():
    fusion = RAGFusionRetriever(
        retriever=FakeRetriever({}),
        query_generator=lambda query: "not a list",
    )

    with pytest.raises(TypeError, match="must return a list"):
        fusion.generate_queries("original query")


def test_generate_queries_rejects_non_string_generated_query():
    fusion, _ = make_fusion({}, ["valid query", 123])

    with pytest.raises(TypeError, match="generated queries must be strings"):
        fusion.generate_queries("original query")


def test_generate_queries_rejects_empty_generated_query():
    fusion, _ = make_fusion({}, ["   "])

    with pytest.raises(ValueError, match="generated queries cannot be empty"):
        fusion.generate_queries("original query")


def test_constructor_rejects_non_callable_generator():
    with pytest.raises(TypeError, match="query_generator must be callable"):
        RAGFusionRetriever(
            retriever=FakeRetriever({}),
            query_generator="not callable",
        )


@pytest.mark.parametrize("retriever", [None, "not callable", 123])
def test_constructor_rejects_invalid_retriever(retriever):
    with pytest.raises(TypeError, match="retriever must be callable"):
        RAGFusionRetriever(
            retriever=retriever,
            query_generator=lambda query: [],
        )


@pytest.mark.parametrize("rrf_constant", [True, False, 1.5, "60"])
def test_constructor_rejects_non_integer_rrf_constant(rrf_constant):
    with pytest.raises(TypeError, match="rrf_constant must be an integer"):
        RAGFusionRetriever(
            retriever=FakeRetriever({}),
            query_generator=lambda query: [],
            rrf_constant=rrf_constant,
        )


@pytest.mark.parametrize("rrf_constant", [0, -1])
def test_constructor_rejects_non_positive_rrf_constant(rrf_constant):
    with pytest.raises(ValueError, match="rrf_constant must be greater than zero"):
        RAGFusionRetriever(
            retriever=FakeRetriever({}),
            query_generator=lambda query: [],
            rrf_constant=rrf_constant,
        )


@pytest.mark.parametrize("k", [True, False, 1.5, "3"])
def test_search_rejects_non_integer_k(k):
    fusion, _ = make_fusion({})

    with pytest.raises(TypeError, match="k must be an integer"):
        fusion.search("original query", k=k)


@pytest.mark.parametrize("k", [0, -1])
def test_search_rejects_non_positive_k(k):
    fusion, _ = make_fusion({})

    with pytest.raises(ValueError, match="k must be greater than zero"):
        fusion.search("original query", k=k)


@pytest.mark.parametrize("query", [None, 123, []])
def test_search_rejects_non_string_query(query):
    fusion, _ = make_fusion({})

    with pytest.raises(TypeError, match="query must be a string"):
        fusion.search(query)


@pytest.mark.parametrize("query", ["", "   "])
def test_search_rejects_empty_query(query):
    fusion, _ = make_fusion({})

    with pytest.raises(ValueError, match="query cannot be empty"):
        fusion.search(query)


def test_search_combines_scores_using_reciprocal_rank_fusion():
    results = {
        "original query": [
            make_result(1),
            make_result(2),
        ],
        "alternate query": [
            make_result(2),
            make_result(3),
        ],
    }
    fusion, _ = make_fusion(results, ["alternate query"], rrf_constant=1)

    ranked = fusion.search("original query", k=3)

    assert [item["index"] for item in ranked] == [2, 1, 3]
    assert ranked[0]["rrf_score"] == pytest.approx(1 / 3 + 1 / 2)
    assert ranked[1]["rrf_score"] == pytest.approx(1 / 2)
    assert ranked[2]["rrf_score"] == pytest.approx(1 / 3)


def test_search_tracks_matched_queries():
    results = {
        "original query": [make_result(1)],
        "alternate query": [make_result(1)],
    }
    fusion, _ = make_fusion(results, ["alternate query"])

    ranked = fusion.search("original query")

    assert ranked[0]["matched_queries"] == [
        "original query",
        "alternate query",
    ]


def test_search_calls_retriever_for_each_unique_query():
    fusion, retriever = make_fusion(
        {},
        ["alternate query", "ALTERNATE QUERY"],
    )

    fusion.search("original query")

    assert retriever.calls == [
        ("original query", 3),
        ("alternate query", 3),
    ]


def test_search_limits_results_to_k():
    results = {
        "original query": [
            make_result(1),
            make_result(2),
            make_result(3),
        ],
        "alternate query": [
            make_result(3),
            make_result(2),
            make_result(1),
        ],
    }
    fusion, _ = make_fusion(results, ["alternate query"])

    assert len(fusion.search("original query", k=2)) == 2


def test_search_rejects_non_list_retriever_results():
    fusion = RAGFusionRetriever(
        retriever=lambda query: "not a list",
        query_generator=lambda query: [],
    )

    with pytest.raises(TypeError, match="retriever must return a list"):
        fusion.search("original query")


def test_search_rejects_non_dictionary_result():
    fusion, _ = make_fusion({"original query": ["not a dictionary"]})

    with pytest.raises(TypeError, match="each retrieval result must be a dictionary"):
        fusion.search("original query")


def test_search_rejects_result_without_index():
    fusion, _ = make_fusion(
        {"original query": [{"document": "Missing index"}]}
    )

    with pytest.raises(ValueError, match="must contain an 'index' field"):
        fusion.search("original query")


def test_search_supports_callable_retriever():
    calls = []

    def retriever(query):
        calls.append(query)
        return [make_result(1)]

    fusion = RAGFusionRetriever(
        retriever=retriever,
        query_generator=lambda query: ["alternate query"],
    )

    results = fusion.search("original query")

    assert len(results) == 1
    assert calls == ["original query", "alternate query"]


def test_search_ignores_duplicate_document_within_one_query():
    results = {
        "original query": [
            make_result(1),
            make_result(1, "Duplicate document"),
        ],
    }
    fusion, _ = make_fusion(results, [])

    ranked = fusion.search("original query")

    assert len(ranked) == 1
    assert ranked[0]["rrf_score"] == pytest.approx(1 / 61)
