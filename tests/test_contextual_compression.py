
import pytest

from src.contextual_compression import ContextualCompressor


@pytest.fixture
def compressor():
    return ContextualCompressor()


@pytest.fixture
def documents():
    return [
        {
            "document": (
                "Python was created by Guido van Rossum. "
                "Python supports exception handling with try and except. "
                "Python is popular for data science."
            ),
            "metadata": {"source": "python.txt"},
            "score": 0.9,
            "index": 0,
        },
        {
            "document": (
                "Machine learning trains models using data. "
                "Neural networks can learn complex patterns."
            ),
            "metadata": {"source": "ml.txt"},
            "score": 0.8,
            "index": 1,
        },
    ]


def test_compress_returns_list(compressor, documents):
    results = compressor.compress("Python exception handling", documents)

    assert isinstance(results, list)


def test_compress_keeps_relevant_sentences(compressor, documents):
    results = compressor.compress("Python exception handling", documents)

    assert len(results) == 1
    assert "exception handling" in results[0]["document"]


def test_compress_removes_irrelevant_sentences(compressor, documents):
    results = compressor.compress("Python exception handling", documents)

    text = results[0]["document"]

    assert "Guido van Rossum" not in text
    assert "data science" not in text


def test_compress_preserves_metadata_and_scores(compressor, documents):
    results = compressor.compress("Python exception handling", documents)

    assert results[0]["metadata"] == {"source": "python.txt"}
    assert results[0]["score"] == 0.9
    assert results[0]["index"] == 0


def test_compress_adds_compression_ratio(compressor, documents):
    results = compressor.compress("Python exception handling", documents)

    assert 0 < results[0]["compression_ratio"] <= 1


def test_compress_respects_top_n(compressor, documents):
    results = compressor.compress(
        "Python exception handling",
        documents,
        top_n=1,
    )

    assert len(results) <= 1


def test_compress_returns_empty_for_no_documents(compressor):
    assert compressor.compress("Python", []) == []


def test_compress_returns_empty_when_no_sentence_matches(compressor):
    docs = [
        {"document": "Cooking recipes use salt and spices."}
    ]

    results = compressor.compress("quantum physics", docs)

    assert results == []


def test_compress_rejects_non_string_query(compressor, documents):
    with pytest.raises(TypeError, match="query must be a string"):
        compressor.compress(123, documents)


def test_compress_rejects_empty_query(compressor, documents):
    with pytest.raises(ValueError, match="query cannot be empty"):
        compressor.compress("   ", documents)


def test_compress_rejects_non_list_documents(compressor):
    with pytest.raises(TypeError, match="documents must be a list"):
        compressor.compress("Python", "not a list")


def test_compress_rejects_invalid_document_item(compressor):
    with pytest.raises(
        TypeError,
        match=r"documents\[0\] must be a dictionary",
    ):
        compressor.compress("Python", ["invalid"])


def test_compress_rejects_missing_document_text(compressor):
    with pytest.raises(
        TypeError,
        match=r"documents\[0\]\['document'\] must be a string",
    ):
        compressor.compress("Python", [{"metadata": {}}])


def test_compress_rejects_empty_document_text(compressor):
    with pytest.raises(
        ValueError,
        match=r"documents\[0\]\['document'\] cannot be empty",
    ):
        compressor.compress("Python", [{"document": "   "}])


@pytest.mark.parametrize("invalid_top_n", [0, -1])
def test_compress_rejects_non_positive_top_n(
    compressor,
    documents,
    invalid_top_n,
):
    with pytest.raises(
        ValueError,
        match="top_n must be greater than zero",
    ):
        compressor.compress(
            "Python",
            documents,
            top_n=invalid_top_n,
        )


@pytest.mark.parametrize("invalid_top_n", [1.5, "2", True])
def test_compress_rejects_invalid_top_n_type(
    compressor,
    documents,
    invalid_top_n,
):
    with pytest.raises(
        TypeError,
        match="top_n must be an integer or None",
    ):
        compressor.compress(
            "Python",
            documents,
            top_n=invalid_top_n,
        )


def test_constructor_rejects_non_integer_min_sentence_length():
    with pytest.raises(
        TypeError,
        match="min_sentence_length must be an integer",
    ):
        ContextualCompressor(min_sentence_length=2.5)


def test_constructor_rejects_non_positive_min_sentence_length():
    with pytest.raises(
        ValueError,
        match="min_sentence_length must be greater than zero",
    ):
        ContextualCompressor(min_sentence_length=0)


def test_compress_does_not_modify_original_documents(compressor, documents):
    original = [item.copy() for item in documents]

    compressor.compress("Python exception handling", documents)

    assert documents == original
