
import pytest

import src.self_query_retriever as self_query_module
from src.self_query_retriever import SelfQueryRetriever


class FakeVectorStore:
    """Small in-memory vector store for isolated unit tests."""

    def __init__(self, dimension):
        self.dimension = dimension
        self.results = []

    def add(self, vectors, documents, metadata):
        self.results = [
            {
                "document": document,
                "metadata": item_metadata,
                "score": 1.0 / (index + 1),
                "index": index,
            }
            for index, (document, item_metadata) in enumerate(
                zip(documents, metadata)
            )
        ]

    def search(self, query_vector, k):
        return self.results[:k]


def make_retriever(monkeypatch, documents):
    """Build a retriever without loading a real embedding model."""
    monkeypatch.setattr(
        self_query_module,
        "embed_texts",
        lambda texts, model: [[0.1, 0.2] for _ in texts],
    )
    monkeypatch.setattr(
        self_query_module,
        "embed_query",
        lambda query, model: [0.1, 0.2],
    )
    monkeypatch.setattr(self_query_module, "VectorStore", FakeVectorStore)

    return SelfQueryRetriever(documents, model=object())


@pytest.fixture
def sample_documents():
    return [
        {
            "document": "Python programming and machine learning.",
            "metadata": {"year": 2025, "category": "python"},
        },
        {
            "document": "Python basics and functions.",
            "metadata": {"year": 2023, "category": "python"},
        },
        {
            "document": "Java application development.",
            "metadata": {"year": 2025, "category": "java"},
        },
        {
            "document": "Python AI engineering.",
            "metadata": {"year": 2026, "category": "python"},
        },
    ]


def test_extract_after_year_filter():
    semantic_query, filters = SelfQueryRetriever.extract_filters(
        "Python documents after 2024"
    )

    assert semantic_query == "Python documents"
    assert filters == {"year": {"operator": ">", "value": 2024}}


def test_extract_before_year_filter():
    semantic_query, filters = SelfQueryRetriever.extract_filters(
        "Python documents before 2025"
    )

    assert semantic_query == "Python documents"
    assert filters == {"year": {"operator": "<", "value": 2025}}


def test_extract_year_greater_than_filter():
    semantic_query, filters = SelfQueryRetriever.extract_filters(
        "Python year > 2024"
    )

    assert semantic_query == "Python"
    assert filters == {"year": {"operator": ">", "value": 2024}}


def test_extract_year_less_than_filter():
    semantic_query, filters = SelfQueryRetriever.extract_filters(
        "Python year < 2025"
    )

    assert semantic_query == "Python"
    assert filters == {"year": {"operator": "<", "value": 2025}}


def test_extract_category_filter():
    semantic_query, filters = SelfQueryRetriever.extract_filters(
        "Python guides category: python"
    )

    assert semantic_query == "Python guides"
    assert filters == {
        "category": {"operator": "==", "value": "python"}
    }


def test_extract_year_and_category_filters():
    semantic_query, filters = SelfQueryRetriever.extract_filters(
        "Python documents after 2024 category: python"
    )

    assert semantic_query == "Python documents"
    assert filters == {
        "year": {"operator": ">", "value": 2024},
        "category": {"operator": "==", "value": "python"},
    }


def test_extract_filters_without_metadata_conditions():
    semantic_query, filters = SelfQueryRetriever.extract_filters(
        "Explain Python programming"
    )

    assert semantic_query == "Explain Python programming"
    assert filters == {}


@pytest.mark.parametrize("query", ["", "   "])
def test_extract_filters_rejects_empty_query(query):
    with pytest.raises(ValueError, match="must not be empty"):
        SelfQueryRetriever.extract_filters(query)


def test_extract_filters_rejects_non_string_query():
    with pytest.raises(TypeError, match="query must be a string"):
        SelfQueryRetriever.extract_filters(None)


def test_extract_filters_rejects_filter_only_query():
    with pytest.raises(ValueError, match="semantic search term"):
        SelfQueryRetriever.extract_filters("after 2024")


def test_constructor_rejects_empty_documents():
    with pytest.raises(ValueError, match="non-empty list"):
        SelfQueryRetriever([], model=object())


def test_constructor_rejects_non_list_documents():
    with pytest.raises(ValueError, match="non-empty list"):
        SelfQueryRetriever("not a list", model=object())


def test_constructor_rejects_non_dictionary_document(monkeypatch):
    with pytest.raises(TypeError, match="each document must be a dictionary"):
        make_retriever(monkeypatch, ["not a dictionary"])


@pytest.mark.parametrize("document", ["", "   ", None, 123])
def test_constructor_rejects_invalid_document_text(monkeypatch, document):
    with pytest.raises(ValueError, match="document must be a non-empty string"):
        make_retriever(
            monkeypatch,
            [{"document": document, "metadata": {}}],
        )


def test_constructor_rejects_non_dictionary_metadata(monkeypatch):
    with pytest.raises(TypeError, match="metadata must be a dictionary"):
        make_retriever(
            monkeypatch,
            [{"document": "Python guide", "metadata": []}],
        )


def test_search_returns_top_k_without_filters(monkeypatch, sample_documents):
    retriever = make_retriever(monkeypatch, sample_documents)

    results = retriever.search("Python programming", k=2)

    assert len(results) == 2
    assert results[0]["document"] == sample_documents[0]["document"]
    assert results[1]["document"] == sample_documents[1]["document"]


def test_search_filters_by_year(monkeypatch, sample_documents):
    retriever = make_retriever(monkeypatch, sample_documents)

    results = retriever.search("Python documents after 2024", k=3)

    assert [item["metadata"]["year"] for item in results] == [
        2025,
        2025,
        2026,
    ]


def test_search_filters_by_category(monkeypatch, sample_documents):
    retriever = make_retriever(monkeypatch, sample_documents)

    results = retriever.search("programming category: python", k=4)

    assert all(item["metadata"]["category"] == "python" for item in results)
    assert len(results) == 3


def test_search_applies_year_and_category_filters(
    monkeypatch, sample_documents
):
    retriever = make_retriever(monkeypatch, sample_documents)

    results = retriever.search(
        "Python documents after 2024 category: python",
        k=4,
    )

    assert [item["metadata"]["year"] for item in results] == [2025, 2026]
    assert all(item["metadata"]["category"] == "python" for item in results)


def test_search_preserves_similarity_order_after_filtering(
    monkeypatch, sample_documents
):
    retriever = make_retriever(monkeypatch, sample_documents)

    results = retriever.search("Python after 2024", k=2)

    assert [item["index"] for item in results] == [0, 2]


def test_search_respects_k_after_filtering(monkeypatch, sample_documents):
    retriever = make_retriever(monkeypatch, sample_documents)

    results = retriever.search("Python category: python", k=2)

    assert len(results) == 2


@pytest.mark.parametrize("query", ["", "   "])
def test_search_rejects_empty_query(monkeypatch, sample_documents, query):
    retriever = make_retriever(monkeypatch, sample_documents)

    with pytest.raises(ValueError, match="must not be empty"):
        retriever.search(query)


def test_search_rejects_non_string_query(monkeypatch, sample_documents):
    retriever = make_retriever(monkeypatch, sample_documents)

    with pytest.raises(TypeError, match="query must be a string"):
        retriever.search(None)


@pytest.mark.parametrize("k", [True, False, 1.5, "2", None])
def test_search_rejects_non_integer_k(monkeypatch, sample_documents, k):
    retriever = make_retriever(monkeypatch, sample_documents)

    with pytest.raises(TypeError, match="k must be an integer"):
        retriever.search("Python", k=k)


@pytest.mark.parametrize("k", [0, -1])
def test_search_rejects_non_positive_k(monkeypatch, sample_documents, k):
    retriever = make_retriever(monkeypatch, sample_documents)

    with pytest.raises(ValueError, match="k must be positive"):
        retriever.search("Python", k=k)


def test_year_filter_rejects_non_numeric_metadata():
    assert not SelfQueryRetriever._matches_filters(
        {"year": "unknown"},
        {"year": {"operator": ">", "value": 2024}},
    )


def test_filter_rejects_missing_metadata_field():
    assert not SelfQueryRetriever._matches_filters(
        {"category": "python"},
        {"year": {"operator": ">", "value": 2024}},
    )


def test_category_filter_is_case_insensitive():
    assert SelfQueryRetriever._matches_filters(
        {"category": "Python"},
        {"category": {"operator": "==", "value": "python"}},
    )


def test_missing_metadata_is_supported(monkeypatch):
    retriever = make_retriever(
        monkeypatch,
        [{"document": "General programming guide"}],
    )

    results = retriever.search("programming", k=1)

    assert len(results) == 1
    assert results[0]["metadata"] == {}
