import pytest

from src.answer_generator import AnswerGenerator


def mock_answer_generator(query: str, context: str) -> str:
    return f"Answer to '{query}' based on: {context}"


def test_generate_returns_answer_and_sources():
    generator = AnswerGenerator(mock_answer_generator)

    documents = [
        {
            "document": "Python is a programming language.",
            "metadata": {"source": "python.txt"},
            "index": 0,
        },
        {
            "document": "Python supports object-oriented programming.",
            "metadata": {"source": "oop.txt"},
            "index": 1,
        },
    ]

    result = generator.generate("What is Python?", documents)

    assert result["query"] == "What is Python?"
    assert "Python is a programming language." in result["answer"]
    assert len(result["sources"]) == 2
    assert result["sources"][0]["index"] == 0
    assert result["sources"][0]["metadata"]["source"] == "python.txt"


def test_generate_passes_query_and_context_to_generator():
    received = {}

    def mock_generator(query: str, context: str) -> str:
        received["query"] = query
        received["context"] = context
        return "Python is a programming language."

    generator = AnswerGenerator(mock_generator)

    generator.generate(
        "What is Python?",
        [{"document": "Python is a programming language."}],
    )

    assert received["query"] == "What is Python?"
    assert "[1] Python is a programming language." in received["context"]


def test_generate_returns_fallback_when_no_documents():
    generator = AnswerGenerator(mock_answer_generator)

    result = generator.generate("Unknown question", [])

    assert result["answer"] == (
        "I could not find relevant information to answer this question."
    )
    assert result["sources"] == []


def test_generate_strips_query_and_answer():
    generator = AnswerGenerator(
        lambda query, context: "  Generated answer.  "
    )

    result = generator.generate(
        "  What is Python?  ",
        [{"document": "Python is a language."}],
    )

    assert result["query"] == "What is Python?"
    assert result["answer"] == "Generated answer."


def test_init_rejects_non_callable_generator():
    with pytest.raises(TypeError, match="must be callable"):
        AnswerGenerator("not callable")


def test_init_rejects_non_string_fallback():
    with pytest.raises(TypeError, match="must be a string"):
        AnswerGenerator(mock_answer_generator, no_context_answer=None)


def test_init_rejects_empty_fallback():
    with pytest.raises(ValueError, match="cannot be empty"):
        AnswerGenerator(mock_answer_generator, no_context_answer="   ")


@pytest.mark.parametrize("query", [None, 123, [], {}])
def test_generate_rejects_non_string_query(query):
    generator = AnswerGenerator(mock_answer_generator)

    with pytest.raises(TypeError, match="query must be a string"):
        generator.generate(query, [])


@pytest.mark.parametrize("query", ["", "   ", "\n"])
def test_generate_rejects_empty_query(query):
    generator = AnswerGenerator(mock_answer_generator)

    with pytest.raises(ValueError, match="query cannot be empty"):
        generator.generate(query, [])


@pytest.mark.parametrize("documents", [None, "document", {}, 123])
def test_generate_rejects_non_list_documents(documents):
    generator = AnswerGenerator(mock_answer_generator)

    with pytest.raises(TypeError, match="must be a list"):
        generator.generate("Question?", documents)


@pytest.mark.parametrize("document", ["invalid", None, 123, []])
def test_generate_rejects_non_dictionary_document(document):
    generator = AnswerGenerator(mock_answer_generator)

    with pytest.raises(TypeError, match="must be a dictionary"):
        generator.generate("Question?", [document])


@pytest.mark.parametrize(
    "document",
    [
        {},
        {"document": None},
        {"document": 123},
        {"document": []},
    ],
)
def test_generate_rejects_non_string_document_text(document):
    generator = AnswerGenerator(mock_answer_generator)

    with pytest.raises(TypeError, match="must contain string text"):
        generator.generate("Question?", [document])


@pytest.mark.parametrize("document", ["", "   ", "\n"])
def test_generate_rejects_empty_document_text(document):
    generator = AnswerGenerator(mock_answer_generator)

    with pytest.raises(ValueError, match="cannot be empty"):
        generator.generate("Question?", [{"document": document}])


def test_generate_rejects_non_dictionary_metadata():
    generator = AnswerGenerator(mock_answer_generator)

    with pytest.raises(TypeError, match="metadata must be a dictionary"):
        generator.generate(
            "Question?",
            [{"document": "Some text.", "metadata": "invalid"}],
        )


def test_generate_rejects_non_string_generated_answer():
    generator = AnswerGenerator(lambda query, context: None)

    with pytest.raises(TypeError, match="must return a string"):
        generator.generate(
            "Question?",
            [{"document": "Some context."}],
        )


@pytest.mark.parametrize("answer", ["", "   ", "\n"])
def test_generate_rejects_empty_generated_answer(answer):
    generator = AnswerGenerator(lambda query, context: answer)

    with pytest.raises(ValueError, match="generated answer cannot be empty"):
        generator.generate(
            "Question?",
            [{"document": "Some context."}],
        )


def test_generate_does_not_call_generator_without_documents():
    called = False

    def mock_generator(query: str, context: str) -> str:
        nonlocal called
        called = True
        return "Answer"

    generator = AnswerGenerator(mock_generator)

    generator.generate("Question?", [])

    assert called is False


def test_generate_preserves_metadata_without_mutating_original():
    metadata = {"source": "notes.txt"}
    documents = [{"document": "Some text.", "metadata": metadata}]

    generator = AnswerGenerator(mock_answer_generator)
    result = generator.generate("Question?", documents)

    assert result["sources"][0]["metadata"] == metadata
    assert result["sources"][0]["metadata"] is not metadata


def test_generate_supports_missing_metadata_and_index():
    generator = AnswerGenerator(mock_answer_generator)

    result = generator.generate(
        "Question?",
        [{"document": "Some text."}],
    )

    assert result["sources"][0]["metadata"] == {}
    assert result["sources"][0]["index"] is None