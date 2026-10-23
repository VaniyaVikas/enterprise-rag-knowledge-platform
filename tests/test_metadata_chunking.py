
import pytest

from src.metadata_chunking import split_text_with_metadata
from src.document_chunk import DocumentChunk


def test_split_text_with_metadata_returns_document_chunks():
    chunks = split_text_with_metadata(
        text="First paragraph.\n\nSecond paragraph.",
        source="sample.txt",
        chunk_size=25,
        chunk_overlap=5,
    )

    assert isinstance(chunks, list)
    assert all(isinstance(chunk, DocumentChunk) for chunk in chunks)


def test_split_text_with_metadata_preserves_source():
    chunks = split_text_with_metadata(
        text="RAG retrieves relevant documents.",
        source="data/raw/guide.txt",
    )

    assert chunks
    assert all(
        chunk.source == "data/raw/guide.txt"
        for chunk in chunks
    )


def test_split_text_with_metadata_assigns_sequential_chunk_ids():
    chunks = split_text_with_metadata(
        text=" ".join(["RAG"] * 100),
        source="sample.txt",
        chunk_size=50,
        chunk_overlap=5,
    )

    assert len(chunks) > 1
    assert [chunk.chunk_id for chunk in chunks] == list(range(len(chunks)))


def test_split_text_with_metadata_exposes_metadata():
    chunks = split_text_with_metadata(
        text="RAG retrieves information.",
        source="sample.txt",
    )

    assert chunks[0].metadata["source"] == "sample.txt"
    assert chunks[0].metadata["chunk_id"] == 0
    assert chunks[0].metadata["chunk_size"] == len(chunks[0].text)


def test_split_text_with_metadata_handles_empty_text():
    chunks = split_text_with_metadata(
        text="",
        source="empty.txt",
    )

    assert chunks == []


def test_split_text_with_metadata_rejects_non_string_source():
    with pytest.raises(TypeError, match="source must be a string."):
        split_text_with_metadata(
            text="Some text",
            source=123,
        )


@pytest.mark.parametrize("source", ["", "   ", "\n"])
def test_split_text_with_metadata_rejects_empty_source(source):
    with pytest.raises(ValueError, match="source cannot be empty."):
        split_text_with_metadata(
            text="Some text",
            source=source,
        )


def test_split_text_with_metadata_supports_custom_separators():
    chunks = split_text_with_metadata(
        text="## First section\nContent A\n## Second section\nContent B",
        source="sections.txt",
        chunk_size=30,
        chunk_overlap=0,
        separators=["\n## ", "\n", " ", ""],
    )

    assert len(chunks) > 1
    assert all(len(chunk.text) <= 30 for chunk in chunks)
    assert all(chunk.source == "sections.txt" for chunk in chunks)
