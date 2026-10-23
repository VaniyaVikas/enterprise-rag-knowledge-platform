import pytest

import src.hyde_retriever as hyde_module
from src.hyde_retriever import HyDERetriever


class FakeVectorStore:
    """In-memory vector store for isolated unit tests."""

    def __init__(self, dimension):
        self.dimension = dimension
        self.results = []
        self.last_query_vector = None
        self.last_k = None

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
        self.last_query_vector = query_vector
        self.last_k = k
        return self.results[:k]


def make_retriever(monkeypatch, documents, generator=None):
    """Build a retriever without loading a real embedding model."""
    monkeypatch.setattr(
        hyde_module,
        "embed_texts",
        lambda texts, model: [[0.1, 0.2] for _ in texts],
    )
    monkeypatch.setattr(
        hyde_module,
        "embed_query",
        lambda text, model: [0.3, 0.4],
    )
    monkeypatch.setattr(hyde_module, "VectorStore", FakeVectorStore)

    if generator is None:
        generator = lambda query: f"Hypothetical answer for: {query}"

    return HyDERetriever(
        documents=documents,
        hypothetical_document_generator=generator,
        model=object(),
    )


@pytest.fixture
def sample_documents():
    return [
        {
            "document": "RAG retrieves relevant knowledge to ground answers.",
            "metadata": {"category": "rag"},
        },
        {
            "document": "Python is widely used for AI development.",
            "metadata": {"category": "python"},
        },
        {
            "document": "Vector databases store embedding vectors.",
            "metadata": {"category": "vector-database"},
        },
    ]


def test_constructor_accepts_valid_documents(monkeypatch, sample_documents):
    retriever = make_retriever(monkeypatch, sample_documents)

    assert retriever.documents == [
        item["document"] for item in sample_documents
    ]
    assert retriever.metadata == [
        item["metadata"] for item in sample_documents
    ]


def test_constructor_rejects_empty_documents():
    with pytest.raises(ValueError, match="non-empty list"):
        HyDERetriever(
            [],
            lambda query: "Hypothetical document",
            model=object(),
        )


def test_constructor_rejects_non_list_documents():
    with pytest.raises(ValueError, match="non-empty list"):
        HyDERetriever(
            "not a list",
            lambda query: "Hypothetical document",
            model=object(),
        )


def test_constructor_rejects_non_callable_generator():
    with pytest.raises(TypeError, match="must be callable"):
        HyDERetriever(
            [{"document": "RAG guide"}],
            "not callable",
            model=object(),
        )


def test_constructor_rejects_non_dictionary_document(monkeypatch):
    with pytest.raises(TypeError, match="each document must be a dictionary"):
        make_retriever(monkeypatch, ["not a dictionary"])


@pytest.mark.parametrize("document", ["", "   ", None, 123])
def test_constructor_rejects_invalid_document_text(monkeypatch, document):
    with pytest.raises(
        ValueError,
        match="document must be a non-empty string",
    ):
        make_retriever(
            monkeypatch,
            [{"document": document, "metadata": {}}],
        )


def test_constructor_rejects_non_dictionary_metadata(monkeypatch):
    with pytest.raises(TypeError, match="metadata must be a dictionary"):
        make_retriever(
            monkeypatch,
            [{"document": "RAG guide", "metadata": []}],
        )


def test_generate_hypothetical_document_uses_generator(monkeypatch):
    received_queries = []

    def generator(query):
        received_queries.append(query)
        return "RAG uses retrieved evidence to ground answers."

    retriever = make_retriever(
        monkeypatch,
        [{"document": "RAG guide"}],
        generator=generator,
    )

    result = retriever.generate_hypothetical_document(
        "How does RAG work?"
    )

    assert result == "RAG uses retrieved evidence to ground answers."
    assert received_queries == ["How does RAG work?"]


def test_generate_hypothetical_document_strips_whitespace(monkeypatch):
    retriever = make_retriever(
        monkeypatch,
        [{"document": "RAG guide"}],
        generator=lambda query: "  hypothetical text  ",
    )

    assert retriever.generate_hypothetical_document("Explain RAG") == (
        "hypothetical text"
    )


@pytest.mark.parametrize("query", ["", "   "])
def test_generate_hypothetical_document_rejects_empty_query(
    monkeypatch, query
):
    retriever = make_retriever(
        monkeypatch,
        [{"document": "RAG guide"}],
    )

    with pytest.raises(ValueError, match="query must not be empty"):
        retriever.generate_hypothetical_document(query)


def test_generate_hypothetical_document_rejects_non_string_query(
    monkeypatch,
):
    retriever = make_retriever(
        monkeypatch,
        [{"document": "RAG guide"}],
    )

    with pytest.raises(TypeError, match="query must be a string"):
        retriever.generate_hypothetical_document(None)


@pytest.mark.parametrize("generated", ["", "   ", None, 123])
def test_generate_hypothetical_document_rejects_invalid_output(
    monkeypatch, generated
):
    retriever = make_retriever(
        monkeypatch,
        [{"document": "RAG guide"}],
        generator=lambda query: generated,
    )

    with pytest.raises(
        ValueError,
        match="hypothetical document must be a non-empty string",
    ):
        retriever.generate_hypothetical_document("Explain RAG")


def test_search_uses_hypothetical_document_embedding(
    monkeypatch, sample_documents
):
    embedded_texts = []

    def fake_embed_query(text, model):
        embedded_texts.append(text)
        return [0.7, 0.8]

    # First build the retriever. This helper installs its default mock.
    retriever = make_retriever(
        monkeypatch,
        sample_documents,
        generator=lambda query: "Generated hypothetical document",
    )

    # Then install our test-specific mock so it is not overwritten.
    monkeypatch.setattr(
        hyde_module,
        "embed_query",
        fake_embed_query,
    )

    results = retriever.search("How does RAG work?", k=2)

    assert embedded_texts == ["Generated hypothetical document"]
    assert retriever.vector_store.last_query_vector == [0.7, 0.8]
    assert len(results) == 2


def test_search_calls_generator_with_original_query(
    monkeypatch, sample_documents
):
    received_queries = []

    def generator(query):
        received_queries.append(query)
        return "Hypothetical RAG explanation"

    retriever = make_retriever(
        monkeypatch,
        sample_documents,
        generator=generator,
    )

    retriever.search("How does RAG work?", k=1)

    assert received_queries == ["How does RAG work?"]


def test_search_returns_top_k_results(monkeypatch, sample_documents):
    retriever = make_retriever(monkeypatch, sample_documents)

    results = retriever.search("Explain RAG", k=2)

    assert len(results) == 2
    assert results[0]["document"] == sample_documents[0]["document"]
    assert results[1]["document"] == sample_documents[1]["document"]


def test_search_preserves_document_metadata(monkeypatch, sample_documents):
    retriever = make_retriever(monkeypatch, sample_documents)

    results = retriever.search("Explain RAG", k=1)

    assert results[0]["metadata"] == {"category": "rag"}
    assert results[0]["index"] == 0
    assert results[0]["score"] == 1.0


def test_search_caps_k_at_document_count(monkeypatch, sample_documents):
    retriever = make_retriever(monkeypatch, sample_documents)

    results = retriever.search("Explain RAG", k=100)

    assert len(results) == len(sample_documents)
    assert retriever.vector_store.last_k == len(sample_documents)


@pytest.mark.parametrize("query", ["", "   "])
def test_search_rejects_empty_query(monkeypatch, sample_documents, query):
    retriever = make_retriever(monkeypatch, sample_documents)

    with pytest.raises(ValueError, match="query must not be empty"):
        retriever.search(query)


def test_search_rejects_non_string_query(monkeypatch, sample_documents):
    retriever = make_retriever(monkeypatch, sample_documents)

    with pytest.raises(TypeError, match="query must be a string"):
        retriever.search(None)


@pytest.mark.parametrize("k", [True, False, 1.5, "2", None])
def test_search_rejects_non_integer_k(monkeypatch, sample_documents, k):
    retriever = make_retriever(monkeypatch, sample_documents)

    with pytest.raises(TypeError, match="k must be an integer"):
        retriever.search("Explain RAG", k=k)


@pytest.mark.parametrize("k", [0, -1])
def test_search_rejects_non_positive_k(monkeypatch, sample_documents, k):
    retriever = make_retriever(monkeypatch, sample_documents)

    with pytest.raises(ValueError, match="k must be positive"):
        retriever.search("Explain RAG", k=k)