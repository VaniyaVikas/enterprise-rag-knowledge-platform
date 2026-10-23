
import pytest

from src.document_processor import (
    process_document_with_metadata,
    process_document_from_config,
)
from src.document_chunk import DocumentChunk


def test_process_document_returns_document_chunks():
    chunks = process_document_with_metadata(
        "data/raw/introduction.txt",
        chunk_size=100,
        overlap=20,
    )

    assert len(chunks) > 0
    assert all(isinstance(chunk, DocumentChunk) for chunk in chunks)


def test_document_chunks_contain_source_and_chunk_ids():
    chunks = process_document_with_metadata(
        "data/raw/introduction.txt",
        chunk_size=100,
        overlap=20,
    )

    assert all(
        chunk.source == "data/raw/introduction.txt"
        for chunk in chunks
    )

    assert [chunk.chunk_id for chunk in chunks] == list(range(len(chunks)))


def test_document_chunks_have_valid_metadata():
    chunks = process_document_with_metadata(
        "data/raw/introduction.txt",
        chunk_size=100,
        overlap=20,
    )

    for chunk in chunks:
        assert chunk.metadata["source"] == chunk.source
        assert chunk.metadata["chunk_id"] == chunk.chunk_id
        assert chunk.metadata["chunk_size"] == len(chunk.text)


def test_document_processor_preserves_original_text():
    chunks = process_document_with_metadata(
        "data/raw/introduction.txt",
        chunk_size=100,
        overlap=20,
    )

    combined_text = "".join(
        chunk.text if index == 0 else chunk.text[20:]
        for index, chunk in enumerate(chunks)
    )

    original_text = (
        "Enterprise RAG Knowledge Platform\n\n"
        "RAG stands for Retrieval-Augmented Generation.\n"
        "RAG systems retrieve relevant information from documents\n"
        "and use that information to generate grounded answers.\n\n"
        "LangChain helps developers build applications powered by language models.\n"
        "Document loading is the first step in a typical RAG pipeline.\n"
    )

    assert combined_text == original_text


def test_process_document_from_config_returns_document_chunks():
    chunks = process_document_from_config(
        "data/raw/introduction.txt",
        "configs/chunking.yaml",
    )

    assert len(chunks) > 0
    assert all(isinstance(chunk, DocumentChunk) for chunk in chunks)


def test_process_document_from_config_uses_yaml_settings():
    chunks = process_document_from_config(
        "data/raw/introduction.txt",
        "configs/chunking.yaml",
    )

    assert all(len(chunk.text) <= 100 for chunk in chunks)


def test_process_document_from_config_rejects_missing_config(tmp_path):
    missing_config = tmp_path / "missing.yaml"

    with pytest.raises(FileNotFoundError):
        process_document_from_config(
            "data/raw/introduction.txt",
            str(missing_config),
        )


def test_process_document_from_config_uses_custom_settings(tmp_path):
    config_file = tmp_path / "custom_chunking.yaml"
    config_file.write_text(
        "chunking:\n  chunk_size: 50\n  overlap: 10\n",
        encoding="utf-8",
    )

    chunks = process_document_from_config(
        "data/raw/introduction.txt",
        str(config_file),
    )

    assert len(chunks) > 0
    assert all(len(chunk.text) <= 50 for chunk in chunks)
    assert chunks[0].metadata["chunk_size"] == len(chunks[0].text)
