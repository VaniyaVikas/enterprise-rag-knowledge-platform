
import pytest

from src.document_chunk import DocumentChunk


def test_document_chunk_stores_basic_data():
    chunk = DocumentChunk(
        text="Hello RAG",
        source="test.txt",
        chunk_id=0,
    )

    assert chunk.text == "Hello RAG"
    assert chunk.source == "test.txt"
    assert chunk.chunk_id == 0


def test_document_chunk_returns_metadata():
    chunk = DocumentChunk(
        text="Hello RAG",
        source="test.txt",
        chunk_id=3,
    )

    assert chunk.metadata == {
        "source": "test.txt",
        "chunk_id": 3,
        "chunk_size": 9,
    }


def test_chunk_size_is_calculated_from_text():
    chunk = DocumentChunk(
        text="ABCDEFGHIJ",
        source="test.txt",
        chunk_id=1,
    )

    assert chunk.metadata["chunk_size"] == 10


def test_empty_text_raises_error():
    with pytest.raises(ValueError, match="Chunk text cannot be empty."):
        DocumentChunk(
            text="",
            source="test.txt",
            chunk_id=0,
        )


def test_empty_source_raises_error():
    with pytest.raises(ValueError, match="Chunk source cannot be empty."):
        DocumentChunk(
            text="Hello RAG",
            source="",
            chunk_id=0,
        )


def test_negative_chunk_id_raises_error():
    with pytest.raises(ValueError, match="chunk_id cannot be negative."):
        DocumentChunk(
            text="Hello RAG",
            source="test.txt",
            chunk_id=-1,
        )


def test_non_string_text_raises_type_error():
    with pytest.raises(TypeError, match="Chunk text must be a string."):
        DocumentChunk(
            text=123,
            source="test.txt",
            chunk_id=0,
        )


def test_non_string_source_raises_type_error():
    with pytest.raises(TypeError, match="Chunk source must be a string."):
        DocumentChunk(
            text="Hello RAG",
            source=123,
            chunk_id=0,
        )


def test_non_integer_chunk_id_raises_type_error():
    with pytest.raises(TypeError, match="chunk_id must be an integer."):
        DocumentChunk(
            text="Hello RAG",
            source="test.txt",
            chunk_id=1.5,
        )


def test_boolean_chunk_id_raises_type_error():
    with pytest.raises(TypeError, match="chunk_id must be an integer."):
        DocumentChunk(
            text="Hello RAG",
            source="test.txt",
            chunk_id=True,
        )
