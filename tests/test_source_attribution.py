
import pytest

from src.source_attribution import SourceAttributor


def test_format_citations_returns_numbered_sources():
    attributor = SourceAttributor()

    citations = attributor.format_citations(
        [
            {
                "document": "Python is a programming language.",
                "metadata": {"source": "python_guide.pdf"},
                "index": 0,
            },
            {
                "document": "Python supports functions.",
                "metadata": {"source": "functions.pdf"},
                "index": 1,
            },
        ]
    )

    assert citations == [
        {
            "citation_id": 1,
            "source": "python_guide.pdf",
            "index": 0,
        },
        {
            "citation_id": 2,
            "source": "functions.pdf",
            "index": 1,
        },
    ]


def test_format_citations_uses_url_when_source_is_missing():
    attributor = SourceAttributor()

    citations = attributor.format_citations(
        [
            {
                "document": "Page content.",
                "metadata": {"url": "https://example.com/page"},
            }
        ]
    )

    assert citations[0]["source"] == "https://example.com/page"


def test_format_citations_uses_unknown_source_fallback():
    attributor = SourceAttributor()

    citations = attributor.format_citations(
        [{"document": "Some content."}]
    )

    assert citations[0]["source"] == "Unknown source"


def test_format_citations_includes_document_when_enabled():
    attributor = SourceAttributor(include_document_text=True)

    citations = attributor.format_citations(
        [{"document": "  Python is useful.  "}]
    )

    assert citations[0]["document"] == "Python is useful."


def test_format_citations_omits_document_by_default():
    attributor = SourceAttributor()

    citations = attributor.format_citations(
        [{"document": "Python is useful."}]
    )

    assert "document" not in citations[0]


def test_attach_citations_returns_answer_and_citations():
    attributor = SourceAttributor()

    result = attributor.attach_citations(
        "  Python is useful.  ",
        [
            {
                "document": "Python is useful.",
                "metadata": {"source": "python.txt"},
                "index": 2,
            }
        ],
    )

    assert result == {
        "answer": "Python is useful.",
        "citations": [
            {
                "citation_id": 1,
                "source": "python.txt",
                "index": 2,
            }
        ],
    }


def test_init_rejects_non_boolean_include_document_text():
    with pytest.raises(TypeError, match="must be a boolean"):
        SourceAttributor(include_document_text="yes")


@pytest.mark.parametrize("sources", [None, "source", {}, 123])
def test_format_citations_rejects_non_list_sources(sources):
    attributor = SourceAttributor()

    with pytest.raises(TypeError, match="sources must be a list"):
        attributor.format_citations(sources)


@pytest.mark.parametrize("source", ["invalid", None, 123, []])
def test_format_citations_rejects_non_dictionary_source(source):
    attributor = SourceAttributor()

    with pytest.raises(TypeError, match="each source must be a dictionary"):
        attributor.format_citations([source])


@pytest.mark.parametrize(
    "source",
    [
        {},
        {"document": None},
        {"document": 123},
        {"document": []},
    ],
)
def test_format_citations_rejects_non_string_document(source):
    attributor = SourceAttributor()

    with pytest.raises(TypeError, match="must contain string document text"):
        attributor.format_citations([source])


@pytest.mark.parametrize("document", ["", "   ", "\n"])
def test_format_citations_rejects_empty_document(document):
    attributor = SourceAttributor()

    with pytest.raises(ValueError, match="source document text cannot be empty"):
        attributor.format_citations([{"document": document}])


def test_format_citations_rejects_non_dictionary_metadata():
    attributor = SourceAttributor()

    with pytest.raises(TypeError, match="source metadata must be a dictionary"):
        attributor.format_citations(
            [{"document": "Content.", "metadata": "invalid"}]
        )


@pytest.mark.parametrize("answer", [None, 123, [], {}])
def test_attach_citations_rejects_non_string_answer(answer):
    attributor = SourceAttributor()

    with pytest.raises(TypeError, match="answer must be a string"):
        attributor.attach_citations(answer, [])


@pytest.mark.parametrize("answer", ["", "   ", "\n"])
def test_attach_citations_rejects_empty_answer(answer):
    attributor = SourceAttributor()

    with pytest.raises(ValueError, match="answer cannot be empty"):
        attributor.attach_citations(answer, [])


def test_format_citations_does_not_mutate_original_metadata():
    metadata = {"source": "guide.txt"}
    sources = [{"document": "Some content.", "metadata": metadata}]

    SourceAttributor().format_citations(sources)

    assert metadata == {"source": "guide.txt"}


def test_format_citations_supports_missing_index():
    attributor = SourceAttributor()

    citations = attributor.format_citations(
        [{"document": "Some content."}]
    )

    assert citations[0] == {
        "citation_id": 1,
        "source": "Unknown source",
    }
