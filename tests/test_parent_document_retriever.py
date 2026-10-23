
import pytest

from src.parent_document_retriever import ParentDocumentRetriever


class FakeEmbeddingModel:
    """Simple marker object for injected-model validation."""


def make_parent_documents():
    return [
        {
            "parent_id": "doc_1",
            "document": (
                "LangChain is a framework for building AI applications. "
                "It supports retrieval augmented generation and agents."
            ),
            "metadata": {"source": "langchain.txt"},
        },
        {
            "parent_id": "doc_2",
            "document": (
                "FAISS is a library for efficient similarity search. "
                "It works with vector embeddings."
            ),
            "metadata": {"source": "faiss.txt"},
        },
    ]


def make_child_chunks():
    return [
        {
            "parent_id": "doc_1",
            "document": "LangChain is a framework for building AI applications.",
        },
        {
            "parent_id": "doc_1",
            "document": "It supports retrieval augmented generation and agents.",
        },
        {
            "parent_id": "doc_2",
            "document": "FAISS is a library for efficient similarity search.",
        },
        {
            "parent_id": "doc_2",
            "document": "It works with vector embeddings.",
        },
    ]


def test_constructor_initializes_empty_retriever():
    retriever = ParentDocumentRetriever()

    assert retriever.embedding_model is None
    assert retriever.vector_store is None
    assert retriever.parent_documents == {}


def test_constructor_accepts_custom_embedding_model():
    model = FakeEmbeddingModel()
    retriever = ParentDocumentRetriever(embedding_model=model)

    assert retriever.embedding_model is model


def test_add_documents_indexes_child_chunks():
    retriever = ParentDocumentRetriever()

    retriever.add_documents(
        parent_documents=make_parent_documents(),
        child_chunks=make_child_chunks(),
    )

    assert retriever.vector_store is not None
    assert retriever.vector_store.index.ntotal == 4


def test_add_documents_stores_parent_documents():
    retriever = ParentDocumentRetriever()

    retriever.add_documents(
        parent_documents=make_parent_documents(),
        child_chunks=make_child_chunks(),
    )

    assert set(retriever.parent_documents) == {"doc_1", "doc_2"}


def test_add_documents_stores_child_parent_metadata():
    retriever = ParentDocumentRetriever()

    retriever.add_documents(
        parent_documents=make_parent_documents(),
        child_chunks=make_child_chunks(),
    )

    metadata = retriever.vector_store.metadata

    assert metadata[0]["parent_id"] == "doc_1"
    assert metadata[1]["parent_id"] == "doc_1"
    assert metadata[2]["parent_id"] == "doc_2"
    assert metadata[3]["parent_id"] == "doc_2"


def test_add_documents_rejects_non_list_parent_documents():
    retriever = ParentDocumentRetriever()

    with pytest.raises(TypeError, match="parent_documents must be a list"):
        retriever.add_documents(
            parent_documents="invalid",
            child_chunks=make_child_chunks(),
        )


def test_add_documents_rejects_non_list_child_chunks():
    retriever = ParentDocumentRetriever()

    with pytest.raises(TypeError, match="child_chunks must be a list"):
        retriever.add_documents(
            parent_documents=make_parent_documents(),
            child_chunks="invalid",
        )


def test_add_documents_rejects_empty_parent_documents():
    retriever = ParentDocumentRetriever()

    with pytest.raises(ValueError, match="parent_documents cannot be empty"):
        retriever.add_documents(
            parent_documents=[],
            child_chunks=make_child_chunks(),
        )


def test_add_documents_rejects_empty_child_chunks():
    retriever = ParentDocumentRetriever()

    with pytest.raises(ValueError, match="child_chunks cannot be empty"):
        retriever.add_documents(
            parent_documents=make_parent_documents(),
            child_chunks=[],
        )


def test_add_documents_rejects_invalid_parent():
    retriever = ParentDocumentRetriever()

    with pytest.raises(
        TypeError,
        match=r"parent_documents\[0\] must be a dictionary",
    ):
        retriever.add_documents(
            parent_documents=["invalid"],
            child_chunks=make_child_chunks(),
        )


def test_add_documents_rejects_non_string_parent_id():
    retriever = ParentDocumentRetriever()

    parents = [
        {
            "parent_id": 123,
            "document": "A valid parent document.",
        }
    ]

    with pytest.raises(
        TypeError,
        match=r"parent_documents\[0\]\['parent_id'\] must be a string",
    ):
        retriever.add_documents(
            parent_documents=parents,
            child_chunks=[
                {
                    "parent_id": "123",
                    "document": "A child chunk.",
                }
            ],
        )


def test_add_documents_rejects_empty_parent_id():
    retriever = ParentDocumentRetriever()

    parents = [
        {
            "parent_id": "   ",
            "document": "Some document.",
        }
    ]

    with pytest.raises(ValueError, match="parent_id.*cannot be empty"):
        retriever.add_documents(
            parent_documents=parents,
            child_chunks=[
                {
                    "parent_id": "doc_1",
                    "document": "Some child.",
                }
            ],
        )


def test_add_documents_rejects_non_string_parent_document():
    retriever = ParentDocumentRetriever()

    parents = [
        {
            "parent_id": "doc_1",
            "document": 123,
        }
    ]

    with pytest.raises(
        TypeError,
        match=r"parent_documents\[0\]\['document'\] must be a string",
    ):
        retriever.add_documents(
            parent_documents=parents,
            child_chunks=[
                {
                    "parent_id": "doc_1",
                    "document": "Some child.",
                }
            ],
        )


def test_add_documents_rejects_empty_parent_document():
    retriever = ParentDocumentRetriever()

    parents = [
        {
            "parent_id": "doc_1",
            "document": "   ",
        }
    ]

    with pytest.raises(
        ValueError,
        match="document.*cannot be empty",
    ):
        retriever.add_documents(
            parent_documents=parents,
            child_chunks=[
                {
                    "parent_id": "doc_1",
                    "document": "Some child.",
                }
            ],
        )


def test_add_documents_rejects_unknown_parent_id():
    retriever = ParentDocumentRetriever()

    children = [
        {
            "parent_id": "unknown",
            "document": "Some child.",
        }
    ]

    with pytest.raises(ValueError, match="Unknown parent_id: unknown"):
        retriever.add_documents(
            parent_documents=make_parent_documents(),
            child_chunks=children,
        )


def test_search_returns_parent_document():
    retriever = ParentDocumentRetriever()

    retriever.add_documents(
        parent_documents=make_parent_documents(),
        child_chunks=make_child_chunks(),
    )

    results = retriever.search(
        query="retrieval augmented generation",
        k=2,
    )

    assert results
    assert results[0]["parent_id"] == "doc_1"
    assert "LangChain is a framework" in results[0]["document"]


def test_search_returns_parent_metadata():
    retriever = ParentDocumentRetriever()

    retriever.add_documents(
        parent_documents=make_parent_documents(),
        child_chunks=make_child_chunks(),
    )

    results = retriever.search(
        query="vector embeddings",
        k=2,
    )

    assert results
    assert results[0]["metadata"] == {"source": "faiss.txt"}


def test_search_does_not_return_duplicate_parent_documents():
    retriever = ParentDocumentRetriever()

    retriever.add_documents(
        parent_documents=make_parent_documents(),
        child_chunks=make_child_chunks(),
    )

    results = retriever.search(
        query="FAISS vector embeddings similarity search",
        k=4,
    )

    parent_ids = [result["parent_id"] for result in results]

    assert len(parent_ids) == len(set(parent_ids))


def test_search_returns_score():
    retriever = ParentDocumentRetriever()

    retriever.add_documents(
        parent_documents=make_parent_documents(),
        child_chunks=make_child_chunks(),
    )

    results = retriever.search(query="FAISS", k=1)

    assert results
    assert isinstance(results[0]["score"], float)


def test_search_returns_child_index():
    retriever = ParentDocumentRetriever()

    retriever.add_documents(
        parent_documents=make_parent_documents(),
        child_chunks=make_child_chunks(),
    )

    results = retriever.search(query="FAISS", k=1)

    assert results
    assert isinstance(results[0]["child_index"], int)


def test_search_rejects_non_string_query():
    retriever = ParentDocumentRetriever()

    with pytest.raises(TypeError, match="query must be a string"):
        retriever.search(123)


def test_search_rejects_empty_query():
    retriever = ParentDocumentRetriever()

    with pytest.raises(ValueError, match="query cannot be empty"):
        retriever.search("   ")


@pytest.mark.parametrize("value", [0, -1])
def test_search_rejects_non_positive_k(value):
    retriever = ParentDocumentRetriever()

    with pytest.raises(ValueError, match="k must be greater than zero"):
        retriever.search("RAG", k=value)


@pytest.mark.parametrize("value", [True, False, 1.5, "3"])
def test_search_rejects_non_integer_k(value):
    retriever = ParentDocumentRetriever()

    with pytest.raises(TypeError, match="k must be an integer"):
        retriever.search("RAG", k=value)


def test_search_rejects_search_before_indexing():
    retriever = ParentDocumentRetriever()

    with pytest.raises(ValueError, match="No documents have been indexed"):
        retriever.search("RAG")


def test_add_documents_rejects_non_dictionary_child():
    retriever = ParentDocumentRetriever()

    with pytest.raises(
        TypeError,
        match=r"child_chunks\[0\] must be a dictionary",
    ):
        retriever.add_documents(
            parent_documents=make_parent_documents(),
            child_chunks=["invalid"],
        )


def test_add_documents_rejects_non_string_child_document():
    retriever = ParentDocumentRetriever()

    children = [
        {
            "parent_id": "doc_1",
            "document": 123,
        }
    ]

    with pytest.raises(
        TypeError,
        match=r"child_chunks\[0\]\['document'\] must be a string",
    ):
        retriever.add_documents(
            parent_documents=make_parent_documents(),
            child_chunks=children,
        )


def test_add_documents_rejects_empty_child_document():
    retriever = ParentDocumentRetriever()

    children = [
        {
            "parent_id": "doc_1",
            "document": "   ",
        }
    ]

    with pytest.raises(ValueError, match="document.*cannot be empty"):
        retriever.add_documents(
            parent_documents=make_parent_documents(),
            child_chunks=children,
        )


def test_add_documents_rejects_non_string_child_parent_id():
    retriever = ParentDocumentRetriever()

    children = [
        {
            "parent_id": 123,
            "document": "Some child.",
        }
    ]

    with pytest.raises(
        TypeError,
        match=r"child_chunks\[0\]\['parent_id'\] must be a string",
    ):
        retriever.add_documents(
            parent_documents=make_parent_documents(),
            child_chunks=children,
        )
