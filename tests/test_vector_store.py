
import numpy as np
import pytest

from src.vector_store import VectorStore


def test_vector_store_initializes_with_correct_dimension():
    store = VectorStore(dimension=3)

    assert store.index.d == 3
    assert store.index.ntotal == 0


@pytest.mark.parametrize("dimension", [0, -1])
def test_vector_store_rejects_non_positive_dimension(dimension):
    with pytest.raises(ValueError, match="dimension must be greater than zero."):
        VectorStore(dimension=dimension)


@pytest.mark.parametrize("dimension", [True, 3.5, "3"])
def test_vector_store_rejects_non_integer_dimension(dimension):
    with pytest.raises(TypeError, match="dimension must be an integer."):
        VectorStore(dimension=dimension)


def test_add_stores_vectors_documents_and_metadata():
    store = VectorStore(dimension=2)

    store.add(
        vectors=[[1.0, 0.0], [0.0, 1.0]],
        documents=["Document A", "Document B"],
        metadata=[{"source": "a.txt"}, {"source": "b.txt"}],
    )

    assert store.index.ntotal == 2
    assert store.documents == ["Document A", "Document B"]
    assert store.metadata == [{"source": "a.txt"}, {"source": "b.txt"}]


def test_add_uses_empty_metadata_when_metadata_is_omitted():
    store = VectorStore(dimension=2)

    store.add(
        vectors=[[1.0, 0.0]],
        documents=["Document A"],
    )

    assert store.metadata == [{}]


def test_add_normalizes_vectors():
    store = VectorStore(dimension=2)

    store.add(
        vectors=[[3.0, 4.0]],
        documents=["Document A"],
    )

    stored_vector = np.zeros((1, 2), dtype=np.float32)
    store.index.reconstruct(0, stored_vector[0])

    assert np.linalg.norm(stored_vector[0]) == pytest.approx(1.0)


def test_add_rejects_mismatched_vector_and_document_counts():
    store = VectorStore(dimension=2)

    with pytest.raises(ValueError, match="vectors and documents must have the same length."):
        store.add(vectors=[[1.0, 0.0]], documents=[])


def test_add_accepts_empty_inputs_without_changing_index():
    store = VectorStore(dimension=2)

    store.add(vectors=[], documents=[])

    assert store.index.ntotal == 0


def test_add_rejects_non_list_vectors():
    store = VectorStore(dimension=2)

    with pytest.raises(TypeError, match="vectors must be a list."):
        store.add(vectors="invalid", documents=["Document A"])


def test_add_rejects_non_list_documents():
    store = VectorStore(dimension=2)

    with pytest.raises(TypeError, match="documents must be a list."):
        store.add(vectors=[[1.0, 0.0]], documents="invalid")


def test_add_rejects_empty_document_text():
    store = VectorStore(dimension=2)

    with pytest.raises(ValueError, match="documents must contain non-empty strings."):
        store.add(vectors=[[1.0, 0.0]], documents=["  "])


def test_add_rejects_wrong_vector_dimensions():
    store = VectorStore(dimension=2)

    with pytest.raises(ValueError, match="vectors must match the configured dimension."):
        store.add(vectors=[[1.0, 0.0, 1.0]], documents=["Document A"])


def test_add_rejects_non_numeric_vectors():
    store = VectorStore(dimension=2)

    with pytest.raises(ValueError, match="vectors must contain valid numeric values."):
        store.add(vectors=[[1.0, "bad"]], documents=["Document A"])


@pytest.mark.parametrize(
    "vector",
    [
        [[float("nan"), 1.0]],
        [[float("inf"), 1.0]],
    ],
)
def test_add_rejects_non_finite_vectors(vector):
    store = VectorStore(dimension=2)

    with pytest.raises(ValueError, match="vectors must contain only finite values."):
        store.add(vectors=vector, documents=["Document A"])


def test_add_rejects_zero_vectors():
    store = VectorStore(dimension=2)

    with pytest.raises(ValueError, match="zero vectors are not supported."):
        store.add(vectors=[[0.0, 0.0]], documents=["Document A"])


def test_add_rejects_mismatched_metadata_count():
    store = VectorStore(dimension=2)

    with pytest.raises(ValueError, match="metadata must match the number of documents."):
        store.add(
            vectors=[[1.0, 0.0]],
            documents=["Document A"],
            metadata=[],
        )


def test_add_rejects_non_dictionary_metadata():
    store = VectorStore(dimension=2)

    with pytest.raises(TypeError, match="each metadata item must be a dictionary."):
        store.add(
            vectors=[[1.0, 0.0]],
            documents=["Document A"],
            metadata=["invalid"],
        )


def test_search_returns_most_similar_document_first():
    store = VectorStore(dimension=2)

    store.add(
        vectors=[[1.0, 0.0], [0.0, 1.0]],
        documents=["Relevant document", "Unrelated document"],
        metadata=[{"source": "relevant.txt"}, {"source": "other.txt"}],
    )

    results = store.search([1.0, 0.0], k=2)

    assert len(results) == 2
    assert results[0]["document"] == "Relevant document"
    assert results[0]["score"] == pytest.approx(1.0)
    assert results[0]["metadata"] == {"source": "relevant.txt"}
    assert results[0]["index"] == 0


def test_search_limits_results_to_k():
    store = VectorStore(dimension=2)

    store.add(
        vectors=[[1.0, 0.0], [0.0, 1.0]],
        documents=["A", "B"],
    )

    assert len(store.search([1.0, 0.0], k=1)) == 1


def test_search_returns_empty_list_for_empty_store():
    store = VectorStore(dimension=2)

    assert store.search([1.0, 0.0]) == []


def test_search_caps_k_at_number_of_stored_vectors():
    store = VectorStore(dimension=2)

    store.add(
        vectors=[[1.0, 0.0]],
        documents=["Only document"],
    )

    assert len(store.search([1.0, 0.0], k=10)) == 1


def test_search_rejects_non_list_query():
    store = VectorStore(dimension=2)

    with pytest.raises(TypeError, match="query_vector must be a list."):
        store.search("invalid")


@pytest.mark.parametrize("k", [0, -1])
def test_search_rejects_non_positive_k(k):
    store = VectorStore(dimension=2)

    with pytest.raises(ValueError, match="k must be greater than zero."):
        store.search([1.0, 0.0], k=k)


@pytest.mark.parametrize("k", [True, 1.5, "2"])
def test_search_rejects_non_integer_k(k):
    store = VectorStore(dimension=2)

    with pytest.raises(TypeError, match="k must be an integer."):
        store.search([1.0, 0.0], k=k)


def test_search_rejects_empty_query():
    store = VectorStore(dimension=2)

    with pytest.raises(ValueError, match="query_vector cannot be empty."):
        store.search([])


def test_search_rejects_wrong_query_dimensions():
    store = VectorStore(dimension=2)

    with pytest.raises(ValueError, match="query_vector must match the configured dimension."):
        store.search([1.0, 0.0, 1.0])


def test_search_rejects_non_numeric_query():
    store = VectorStore(dimension=2)

    with pytest.raises(ValueError, match="query_vector must contain valid numeric values."):
        store.search([1.0, "bad"])


@pytest.mark.parametrize(
    "query",
    [
        [float("nan"), 1.0],
        [float("inf"), 1.0],
    ],
)
def test_search_rejects_non_finite_query(query):
    store = VectorStore(dimension=2)

    with pytest.raises(ValueError, match="query_vector must contain only finite values."):
        store.search(query)


def test_search_rejects_zero_query():
    store = VectorStore(dimension=2)

    with pytest.raises(ValueError, match="zero query vectors are not supported."):
        store.search([0.0, 0.0])
