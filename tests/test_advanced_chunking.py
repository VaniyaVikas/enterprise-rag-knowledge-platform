
import pytest

from src.advanced_chunking import (
    DEFAULT_SEPARATORS,
    split_text_recursively,
)


def test_split_text_recursively_returns_list_of_strings():
    text = "First paragraph.\n\nSecond paragraph."

    chunks = split_text_recursively(
        text,
        chunk_size=25,
        chunk_overlap=5,
    )

    assert isinstance(chunks, list)
    assert all(isinstance(chunk, str) for chunk in chunks)


def test_split_text_recursively_respects_chunk_size():
    text = " ".join(["RAG"] * 100)

    chunks = split_text_recursively(
        text,
        chunk_size=50,
        chunk_overlap=5,
    )

    assert len(chunks) > 1
    assert all(len(chunk) <= 50 for chunk in chunks)


def test_split_text_recursively_handles_empty_text():
    assert split_text_recursively("") == []


def test_split_text_recursively_handles_whitespace_only_text():
    assert split_text_recursively("   \n\n   ") == []


def test_split_text_recursively_rejects_non_string_input():
    with pytest.raises(TypeError, match="text must be a string."):
        split_text_recursively(123)


@pytest.mark.parametrize("chunk_size", [0, -1])
def test_split_text_recursively_rejects_invalid_chunk_size(chunk_size):
    with pytest.raises(
        ValueError,
        match="chunk_size must be greater than zero.",
    ):
        split_text_recursively("Some text", chunk_size=chunk_size)


def test_split_text_recursively_rejects_negative_overlap():
    with pytest.raises(
        ValueError,
        match="chunk_overlap cannot be negative.",
    ):
        split_text_recursively("Some text", chunk_overlap=-1)


def test_split_text_recursively_rejects_overlap_equal_to_chunk_size():
    with pytest.raises(
        ValueError,
        match="chunk_overlap must be smaller than chunk_size.",
    ):
        split_text_recursively(
            "Some text",
            chunk_size=10,
            chunk_overlap=10,
        )


def test_split_text_recursively_preserves_all_text_content():
    text = "RAG retrieves documents.\n\nLangChain builds applications."

    chunks = split_text_recursively(
        text,
        chunk_size=100,
        chunk_overlap=10,
    )

    assert len(chunks) == 1
    assert chunks[0] == text


def test_default_separators_are_configured():
    assert DEFAULT_SEPARATORS == ["\n\n", "\n", " ", ""]


def test_custom_separators_are_supported():
    text = "## First section\nContent A\n## Second section\nContent B"

    chunks = split_text_recursively(
        text,
        chunk_size=30,
        chunk_overlap=0,
        separators=["\n## ", "\n", " ", ""],
    )

    assert len(chunks) > 1
    assert all(len(chunk) <= 30 for chunk in chunks)
    assert any("First section" in chunk for chunk in chunks)
    assert any("Second section" in chunk for chunk in chunks)


def test_empty_separators_are_rejected():
    with pytest.raises(ValueError, match="separators cannot be empty."):
        split_text_recursively("Some text", separators=[])


def test_non_string_separator_is_rejected():
    with pytest.raises(
        TypeError,
        match="all separators must be strings.",
    ):
        split_text_recursively("Some text", separators=["\n", 123])


def test_none_separators_use_default_strategy():
    text = "First paragraph.\n\nSecond paragraph."

    chunks = split_text_recursively(
        text,
        chunk_size=25,
        chunk_overlap=5,
        separators=None,
    )

    assert len(chunks) > 1
    assert all(len(chunk) <= 25 for chunk in chunks)
