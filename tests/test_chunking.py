import pytest

from src.chunking import split_text


def test_split_text_creates_expected_chunks():
    chunks = split_text(
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
        chunk_size=10,
        overlap=2,
    )

    assert chunks == [
        "ABCDEFGHIJ",
        "IJKLMNOPQR",
        "QRSTUVWXYZ",
        "YZ",
    ]


def test_empty_text_returns_empty_list():
    assert split_text("") == []


def test_chunk_size_must_be_positive():
    with pytest.raises(ValueError):
        split_text("Hello", chunk_size=0, overlap=0)


def test_overlap_cannot_be_negative():
    with pytest.raises(ValueError):
        split_text("Hello", chunk_size=5, overlap=-1)


def test_overlap_must_be_smaller_than_chunk_size():
    with pytest.raises(ValueError):
        split_text("Hello", chunk_size=5, overlap=5)
