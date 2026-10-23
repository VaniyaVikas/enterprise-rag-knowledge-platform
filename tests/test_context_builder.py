
import pytest

from src.context_builder import ContextBuilder


def test_build_formats_numbered_context():
    builder = ContextBuilder()

    result = builder.build(
        [
            {"document": "Python is a programming language."},
            {"document": "Python supports multiple paradigms."},
        ]
    )

    assert result == (
        "[1] Python is a programming language.\n\n"
        "[2] Python supports multiple paradigms."
    )


def test_build_returns_empty_string_for_no_documents():
    builder = ContextBuilder()

    assert builder.build([]) == ""


def test_build_strips_document_whitespace():
    builder = ContextBuilder()

    result = builder.build([{"document": "  Python is useful.  "}])

    assert result == "[1] Python is useful."


def test_build_limits_number_of_documents():
    builder = ContextBuilder(max_documents=1)

    result = builder.build(
        [
            {"document": "First document."},
            {"document": "Second document."},
        ]
    )

    assert result == "[1] First document."
    assert "Second document." not in result


def test_build_includes_metadata_when_enabled():
    builder = ContextBuilder(include_metadata=True)

    result = builder.build(
        [
            {
                "document": "Python is useful.",
                "metadata": {"source": "python.txt"},
            }
        ]
    )

    assert "Metadata: source: python.txt" in result
    assert "Content: Python is useful." in result


def test_build_excludes_metadata_by_default():
    builder = ContextBuilder()

    result = builder.build(
        [
            {
                "document": "Python is useful.",
                "metadata": {"source": "python.txt"},
            }
        ]
    )

    assert "python.txt" not in result
    assert "Python is useful." in result


def test_build_limits_total_characters():
    builder = ContextBuilder(max_characters=10)

    result = builder.build([{"document": "A long document."}])

    assert len(result) == 10


def test_build_stops_when_character_limit_is_reached():
    builder = ContextBuilder(max_characters=15)

    result = builder.build(
        [
            {"document": "First document."},
            {"document": "Second document."},
        ]
    )

    assert len(result) <= 15
    assert "Second document." not in result


@pytest.mark.parametrize("max_documents", [0, -1])
def test_init_rejects_non_positive_max_documents(max_documents):
    with pytest.raises(ValueError, match="must be positive"):
        ContextBuilder(max_documents=max_documents)


@pytest.mark.parametrize("max_documents", [True, 1.5, "2"])
def test_init_rejects_invalid_max_documents_type(max_documents):
    with pytest.raises(TypeError, match="must be an integer or None"):
        ContextBuilder(max_documents=max_documents)


@pytest.mark.parametrize("max_characters", [0, -1])
def test_init_rejects_non_positive_max_characters(max_characters):
    with pytest.raises(ValueError, match="must be positive"):
        ContextBuilder(max_characters=max_characters)


@pytest.mark.parametrize("max_characters", [True, 1.5, "20"])
def test_init_rejects_invalid_max_characters_type(max_characters):
    with pytest.raises(TypeError, match="must be an integer or None"):
        ContextBuilder(max_characters=max_characters)


@pytest.mark.parametrize("include_metadata", [1, "yes", None])
def test_init_rejects_invalid_include_metadata(include_metadata):
    with pytest.raises(TypeError, match="must be a boolean"):
        ContextBuilder(include_metadata=include_metadata)


@pytest.mark.parametrize("documents", [None, "text", {}, 123])
def test_build_rejects_non_list_documents(documents):
    builder = ContextBuilder()

    with pytest.raises(TypeError, match="documents must be a list"):
        builder.build(documents)


@pytest.mark.parametrize("document", ["invalid", None, 123, []])
def test_build_rejects_non_dictionary_document(document):
    builder = ContextBuilder()

    with pytest.raises(TypeError, match="each document must be a dictionary"):
        builder.build([document])


@pytest.mark.parametrize(
    "document",
    [
        {},
        {"document": None},
        {"document": 123},
        {"document": []},
    ],
)
def test_build_rejects_non_string_document_text(document):
    builder = ContextBuilder()

    with pytest.raises(TypeError, match="must contain string text"):
        builder.build([document])


@pytest.mark.parametrize("document", ["", "   ", "\n"])
def test_build_rejects_empty_document_text(document):
    builder = ContextBuilder()

    with pytest.raises(ValueError, match="document text cannot be empty"):
        builder.build([{"document": document}])


def test_build_rejects_non_dictionary_metadata():
    builder = ContextBuilder()

    with pytest.raises(TypeError, match="metadata must be a dictionary"):
        builder.build(
            [{"document": "Some text.", "metadata": "invalid"}]
        )


def test_build_preserves_original_metadata():
    metadata = {"source": "notes.txt"}
    documents = [{"document": "Some text.", "metadata": metadata}]

    ContextBuilder(include_metadata=True).build(documents)

    assert metadata == {"source": "notes.txt"}
