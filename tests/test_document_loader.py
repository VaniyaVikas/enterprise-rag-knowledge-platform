import pytest

from src.document_loader import load_text_file


def test_load_existing_text_file():
    content = load_text_file("data/raw/introduction.txt")

    assert "Enterprise RAG Knowledge Platform" in content
    assert "Retrieval-Augmented Generation" in content


def test_missing_file_raises_error():
    with pytest.raises(FileNotFoundError):
        load_text_file("data/raw/does_not_exist.txt")


def test_unsupported_extension_raises_error():
    with pytest.raises(ValueError):
        load_text_file("data/raw/introduction.pdf")
