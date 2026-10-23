
import pytest

from src.contextual_retrieval import ContextualRetriever


def make_retriever(generator=None):
    """Create a retriever using a predictable context generator."""
    if generator is None:
        generator = lambda document, chunk: (
            "This chunk belongs to the document about RAG."
        )

    return ContextualRetriever(context_generator=generator)


def test_constructor_accepts_callable_generator():
    retriever = make_retriever()

    assert callable(retriever.context_generator)


def test_constructor_rejects_non_callable_generator():
    with pytest.raises(TypeError, match="must be callable"):
        ContextualRetriever(context_generator="not callable")


def test_enrich_chunk_adds_context_before_original_chunk():
    retriever = make_retriever()

    result = retriever.enrich_chunk(
        document="The full document explains RAG.",
        chunk="RAG retrieves relevant information.",
    )

    assert result == (
        "This chunk belongs to the document about RAG.\n\n"
        "RAG retrieves relevant information."
    )


def test_enrich_chunk_passes_document_and_chunk_to_generator():
    received = []

    def generator(document, chunk):
        received.append((document, chunk))
        return "Generated context."

    retriever = make_retriever(generator)

    retriever.enrich_chunk(
        "Full source document.",
        "A source chunk.",
    )

    assert received == [
        ("Full source document.", "A source chunk.")
    ]


def test_enrich_chunk_strips_context_and_chunk_whitespace():
    retriever = make_retriever(
        lambda document, chunk: "  Context text.  "
    )

    result = retriever.enrich_chunk(
        "Full document.",
        "  Chunk text.  ",
    )

    assert result == "Context text.\n\nChunk text."


@pytest.mark.parametrize("document", ["", "   "])
def test_enrich_chunk_rejects_empty_document(document):
    retriever = make_retriever()

    with pytest.raises(ValueError, match="document must not be empty"):
        retriever.enrich_chunk(document, "Valid chunk.")


def test_enrich_chunk_rejects_non_string_document():
    retriever = make_retriever()

    with pytest.raises(TypeError, match="document must be a string"):
        retriever.enrich_chunk(None, "Valid chunk.")


@pytest.mark.parametrize("chunk", ["", "   "])
def test_enrich_chunk_rejects_empty_chunk(chunk):
    retriever = make_retriever()

    with pytest.raises(ValueError, match="chunk must not be empty"):
        retriever.enrich_chunk("Valid document.", chunk)


def test_enrich_chunk_rejects_non_string_chunk():
    retriever = make_retriever()

    with pytest.raises(TypeError, match="chunk must be a string"):
        retriever.enrich_chunk("Valid document.", None)


def test_enrich_chunk_rejects_non_string_generated_context():
    retriever = make_retriever(
        lambda document, chunk: None
    )

    with pytest.raises(TypeError, match="generated context must be a string"):
        retriever.enrich_chunk("Valid document.", "Valid chunk.")


@pytest.mark.parametrize("context", ["", "   "])
def test_enrich_chunk_rejects_empty_generated_context(context):
    retriever = make_retriever(
        lambda document, chunk: context
    )

    with pytest.raises(
        ValueError,
        match="generated context must not be empty",
    ):
        retriever.enrich_chunk("Valid document.", "Valid chunk.")


def test_enrich_chunks_processes_multiple_chunks():
    retriever = make_retriever()

    results = retriever.enrich_chunks(
        document="Full RAG document.",
        chunks=[
            "RAG retrieves information.",
            "RAG grounds generated answers.",
        ],
    )

    assert len(results) == 2
    assert results[0].endswith("RAG retrieves information.")
    assert results[1].endswith("RAG grounds generated answers.")


def test_enrich_chunks_rejects_non_list_chunks():
    retriever = make_retriever()

    with pytest.raises(TypeError, match="chunks must be a list"):
        retriever.enrich_chunks("Full document.", "not a list")


def test_enrich_chunks_rejects_empty_list():
    retriever = make_retriever()

    with pytest.raises(ValueError, match="chunks must not be empty"):
        retriever.enrich_chunks("Full document.", [])
