
import pytest

from src.bm25_retriever import BM25Retriever


@pytest.fixture
def documents():
    return [
        "Python is a popular programming language",
        "Python supports programming and automation",
        "Machine learning uses data to train models",
        "Cats and dogs are common pets",
    ]


@pytest.fixture
def retriever(documents):
    return BM25Retriever(documents)


def test_retriever_initializes_with_documents(documents):
    retriever = BM25Retriever(documents)

    assert retriever.documents == documents
    assert len(retriever.tokenized_documents) == len(documents)


def test_search_returns_list_of_results(retriever):
    results = retriever.search("Python programming")

    assert isinstance(results, list)
    assert results


def test_search_results_contain_expected_keys(retriever):
    results = retriever.search("Python")

    assert set(results[0]) == {"document", "score", "index"}


def test_search_ranks_relevant_document_first(retriever):
    results = retriever.search("Python programming")

    assert results[0]["document"] in [
        "Python is a popular programming language",
        "Python supports programming and automation",
    ]


def test_search_respects_k(retriever):
    results = retriever.search("Python", k=2)

    assert len(results) == 2


def test_search_returns_all_documents_when_k_is_large(retriever):
    results = retriever.search("Python", k=10)

    assert len(results) == 4


def test_search_scores_are_floats(retriever):
    results = retriever.search("Python")

    assert all(isinstance(result["score"], float) for result in results)


def test_search_indices_are_valid(retriever):
    results = retriever.search("Python")

    assert all(0 <= result["index"] < 4 for result in results)


def test_search_rejects_non_string_query(retriever):
    with pytest.raises(TypeError, match="query must be a string"):
        retriever.search(123)


def test_search_rejects_empty_query(retriever):
    with pytest.raises(ValueError, match="query cannot be empty"):
        retriever.search("   ")


@pytest.mark.parametrize("invalid_k", [0, -1])
def test_search_rejects_non_positive_k(retriever, invalid_k):
    with pytest.raises(ValueError, match="k must be greater than zero"):
        retriever.search("Python", k=invalid_k)


def test_search_rejects_non_integer_k(retriever):
    with pytest.raises(TypeError, match="k must be an integer"):
        retriever.search("Python", k=1.5)


def test_constructor_rejects_non_list_documents():
    with pytest.raises(TypeError, match="documents must be a list"):
        BM25Retriever("not a list")


def test_constructor_rejects_empty_documents():
    with pytest.raises(ValueError, match="documents cannot be empty"):
        BM25Retriever([])


def test_constructor_rejects_empty_document_text():
    with pytest.raises(ValueError, match="non-empty strings"):
        BM25Retriever(["Python", "   "])


def test_constructor_rejects_non_string_document():
    with pytest.raises(ValueError, match="non-empty strings"):
        BM25Retriever(["Python", 123])
