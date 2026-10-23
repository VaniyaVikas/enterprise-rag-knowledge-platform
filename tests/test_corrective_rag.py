
import pytest

from src.corrective_rag import CorrectiveRAG


class FakeRetriever:
    """Fake retriever that returns configured results."""

    def __init__(self, results):
        self.results = results
        self.queries = []
        self.k_values = []

    def search(self, query, k=3):
        self.queries.append(query)
        self.k_values.append(k)
        return self.results


class FakeCompressor:
    """Fake compressor that returns configured compressed results."""

    def __init__(self, results=None):
        self.results = results
        self.queries = []
        self.documents = []

    def compress(self, query, documents):
        self.queries.append(query)
        self.documents.append(documents)

        if self.results is not None:
            return self.results

        return documents


def make_relevant_document(text="Python is a popular programming language."):
    return {"document": text, "metadata": {"source": "test"}}


def test_constructor_accepts_valid_dependencies():
    retriever = FakeRetriever([make_relevant_document()])
    compressor = FakeCompressor()

    crag = CorrectiveRAG(
        retriever,
        compressor=compressor,
        min_relevant_documents=2,
    )

    assert crag.retriever is retriever
    assert crag.compressor is compressor
    assert crag.min_relevant_documents == 2


def test_constructor_creates_default_compressor():
    crag = CorrectiveRAG(FakeRetriever([]))

    assert crag.compressor is not None
    assert callable(crag.compressor.compress)


@pytest.mark.parametrize("retriever", [None, object(), "invalid"])
def test_constructor_rejects_invalid_retriever(retriever):
    with pytest.raises(
        TypeError,
        match="retriever must be callable or provide a callable search method",
    ):
        CorrectiveRAG(retriever)


@pytest.mark.parametrize("compressor", [object(), "invalid", 123])
def test_constructor_rejects_invalid_compressor(compressor):
    with pytest.raises(
        TypeError,
        match="compressor must provide a callable compress method",
    ):
        CorrectiveRAG(FakeRetriever([]), compressor=compressor)


@pytest.mark.parametrize("fallback", [object(), "invalid", 123])
def test_constructor_rejects_invalid_fallback_retriever(fallback):
    with pytest.raises(
        TypeError,
        match="fallback_retriever must be callable",
    ):
        CorrectiveRAG(
            FakeRetriever([]),
            fallback_retriever=fallback,
        )


@pytest.mark.parametrize(
    "count",
    [True, False, 1.5, "2", None],
)
def test_constructor_rejects_non_integer_threshold(count):
    with pytest.raises(
        TypeError,
        match="min_relevant_documents must be an integer",
    ):
        CorrectiveRAG(
            FakeRetriever([]),
            min_relevant_documents=count,
        )


@pytest.mark.parametrize("count", [0, -1, -3])
def test_constructor_rejects_non_positive_threshold(count):
    with pytest.raises(
        ValueError,
        match="min_relevant_documents must be at least 1",
    ):
        CorrectiveRAG(
            FakeRetriever([]),
            min_relevant_documents=count,
        )


@pytest.mark.parametrize("query", ["", "   ", "\n\t"])
def test_run_rejects_empty_query(query):
    crag = CorrectiveRAG(FakeRetriever([]))

    with pytest.raises(ValueError, match="query must not be empty"):
        crag.run(query)


def test_run_rejects_non_string_query():
    crag = CorrectiveRAG(FakeRetriever([]))

    with pytest.raises(TypeError, match="query must be a string"):
        crag.run(None)


@pytest.mark.parametrize("k", [True, False, 1.5, "3", None])
def test_run_rejects_non_integer_k(k):
    crag = CorrectiveRAG(FakeRetriever([]))

    with pytest.raises(TypeError, match="k must be an integer"):
        crag.run("Python", k=k)


@pytest.mark.parametrize("k", [0, -1, -3])
def test_run_rejects_non_positive_k(k):
    crag = CorrectiveRAG(FakeRetriever([]))

    with pytest.raises(ValueError, match="k must be at least 1"):
        crag.run("Python", k=k)


def test_run_returns_relevant_primary_results_without_fallback():
    documents = [make_relevant_document()]
    retriever = FakeRetriever(documents)
    compressor = FakeCompressor(documents)
    fallback = FakeRetriever([make_relevant_document("Fallback document.")])

    crag = CorrectiveRAG(
        retriever,
        compressor=compressor,
        fallback_retriever=fallback,
    )

    result = crag.run("Python", k=2)

    assert result["query"] == "Python"
    assert result["original_results"] == documents
    assert result["relevant_results"] == documents
    assert result["relevant_count"] == 1
    assert result["fallback_used"] is False
    assert fallback.queries == []


def test_run_uses_fallback_when_primary_has_no_relevant_results():
    primary = FakeRetriever([make_relevant_document("Unrelated content.")])
    fallback_documents = [make_relevant_document()]
    fallback = FakeRetriever(fallback_documents)
    compressor = FakeCompressor(results=[])

    # The fake compressor returns an empty list for both primary and fallback.
    crag = CorrectiveRAG(
        primary,
        compressor=compressor,
        fallback_retriever=fallback,
    )

    result = crag.run("Python")

    assert result["fallback_used"] is True
    assert fallback.queries == ["Python"]
    assert result["relevant_results"] == []


def test_run_passes_query_and_documents_to_compressor():
    documents = [make_relevant_document()]
    retriever = FakeRetriever(documents)
    compressor = FakeCompressor(documents)
    crag = CorrectiveRAG(retriever, compressor=compressor)

    crag.run("  Python  ")

    assert compressor.queries == ["Python"]
    assert compressor.documents == [documents]


def test_run_strips_query_before_retrieval():
    retriever = FakeRetriever([make_relevant_document()])
    compressor = FakeCompressor([make_relevant_document()])
    crag = CorrectiveRAG(retriever, compressor=compressor)

    result = crag.run("  Python  ")

    assert result["query"] == "Python"
    assert retriever.queries == ["Python"]


def test_run_passes_k_to_primary_retriever():
    retriever = FakeRetriever([make_relevant_document()])
    crag = CorrectiveRAG(
        retriever,
        compressor=FakeCompressor([make_relevant_document()]),
    )

    crag.run("Python", k=5)

    assert retriever.k_values == [5]


def test_run_returns_expected_response_structure():
    documents = [make_relevant_document()]
    retriever = FakeRetriever(documents)
    crag = CorrectiveRAG(
        retriever,
        compressor=FakeCompressor(documents),
    )

    result = crag.run("Python")

    assert result == {
        "query": "Python",
        "original_results": documents,
        "relevant_results": documents,
        "relevant_count": 1,
        "fallback_used": False,
    }


def test_run_supports_direct_callable_primary_retriever():
    received_queries = []

    def retriever(query):
        received_queries.append(query)
        return [make_relevant_document()]

    documents = [make_relevant_document()]
    crag = CorrectiveRAG(
        retriever,
        compressor=FakeCompressor(documents),
    )

    result = crag.run("Python")

    assert received_queries == ["Python"]
    assert result["relevant_results"] == documents


def test_run_supports_direct_callable_fallback_retriever():
    primary = FakeRetriever([])
    fallback_queries = []
    fallback_documents = [make_relevant_document()]

    def fallback(query):
        fallback_queries.append(query)
        return fallback_documents

    compressor = FakeCompressor(results=[])
    crag = CorrectiveRAG(
        primary,
        compressor=compressor,
        fallback_retriever=fallback,
    )

    result = crag.run("Python")

    assert fallback_queries == ["Python"]
    assert result["fallback_used"] is True
    assert result["relevant_results"] == []


def test_run_uses_default_compressor_to_filter_unrelated_documents():
    documents = [
        {
            "document": (
                "Python is a popular programming language. "
                "The ocean contains many kinds of fish."
            ),
            "metadata": {"source": "test"},
        }
    ]
    crag = CorrectiveRAG(FakeRetriever(documents))

    result = crag.run("Python programming language")

    assert result["relevant_count"] == 1
    assert "Python is a popular programming language." in (
        result["relevant_results"][0]["document"]
    )
    assert "ocean" not in result["relevant_results"][0]["document"]
