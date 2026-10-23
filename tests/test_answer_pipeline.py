
import pytest

from src.answer_pipeline import AnswerPipeline


def test_pipeline_generates_grounded_answer_with_citations():
    def generator(query: str, prompt: str) -> str:
        assert "Python is a programming language." in prompt
        return "Python is a programming language."

    pipeline = AnswerPipeline(generator)
    result = pipeline.run(
        "What is Python?",
        [
            {
                "document": "Python is a programming language.",
                "metadata": {"source": "python-docs.txt"},
                "index": 0,
            }
        ],
    )

    assert result["query"] == "What is Python?"
    assert result["answer"] == "Python is a programming language."
    assert result["is_valid"] is True
    assert result["action"] == "accept"
    assert result["grounding"]["is_grounded"] is True
    assert result["citations"][0]["source"] == "python-docs.txt"


def test_pipeline_returns_fallback_without_documents():
    pipeline = AnswerPipeline(lambda query, prompt: "Unused answer")
    result = pipeline.run("What is Python?", [])

    assert result["is_valid"] is True
    assert result["action"] == "fallback"
    assert result["reason"] == "no_context"
    assert result["citations"] == []


def test_pipeline_rejects_invalid_query():
    pipeline = AnswerPipeline(lambda query, prompt: "Answer")

    with pytest.raises(ValueError, match="query cannot be empty"):
        pipeline.run("   ", [])


def test_pipeline_rejects_non_string_query():
    pipeline = AnswerPipeline(lambda query, prompt: "Answer")

    with pytest.raises(TypeError, match="query must be a string"):
        pipeline.run(123, [])


def test_pipeline_rejects_non_list_documents():
    pipeline = AnswerPipeline(lambda query, prompt: "Answer")

    with pytest.raises(
        TypeError, match="retrieved_documents must be a list"
    ):
        pipeline.run("Question?", "not a list")


def test_pipeline_rejects_invalid_document():
    pipeline = AnswerPipeline(lambda query, prompt: "Answer")

    with pytest.raises(
        TypeError, match="each retrieved document must be a dictionary"
    ):
        pipeline.run("Question?", ["not a document"])


def test_pipeline_rejects_empty_document_text():
    pipeline = AnswerPipeline(lambda query, prompt: "Answer")

    with pytest.raises(
        ValueError, match="retrieved document text cannot be empty"
    ):
        pipeline.run("Question?", [{"document": "  "}])


def test_pipeline_refuses_unsupported_answer():
    pipeline = AnswerPipeline(
        lambda query, prompt: "The moon is made of cheese."
    )
    result = pipeline.run(
        "What is Python?",
        [{"document": "Python is a programming language."}],
    )

    assert result["is_valid"] is False
    assert result["reason"] == "unsupported_answer"
    assert result["action"] == "refuse"
    assert result["citations"] == []


def test_pipeline_refuses_empty_generated_answer():
    pipeline = AnswerPipeline(lambda query, prompt: "   ")
    result = pipeline.run(
        "Question?",
        [{"document": "Some supporting context."}],
    )

    assert result["is_valid"] is False
    assert result["reason"] == "empty_answer"
    assert result["action"] == "refuse"
    assert result["citations"] == []


def test_pipeline_rejects_non_callable_generator():
    with pytest.raises(
        TypeError, match="answer_generator must be callable"
    ):
        AnswerPipeline(None)


def test_pipeline_does_not_mutate_input_documents():
    documents = [
        {
            "document": "Python is useful.",
            "metadata": {"source": "guide.txt"},
            "index": 7,
        }
    ]
    original = [
        {
            "document": "Python is useful.",
            "metadata": {"source": "guide.txt"},
            "index": 7,
        }
    ]

    pipeline = AnswerPipeline(lambda query, prompt: "Python is useful.")
    pipeline.run("Tell me about Python", documents)

    assert documents == original


def test_pipeline_uses_custom_fallback_validator():
    from src.answer_validator import AnswerValidator

    validator = AnswerValidator(
        no_context_answer="No documents available."
    )
    pipeline = AnswerPipeline(
        lambda query, prompt: "Unused",
        answer_validator=validator,
    )
    result = pipeline.run("Question?", [])

    assert result["answer"] == "No documents available."


def test_pipeline_includes_query_in_prompt():
    captured = {}

    def generator(query: str, prompt: str) -> str:
        captured["prompt"] = prompt
        return "Python is useful."

    pipeline = AnswerPipeline(generator)
    pipeline.run(
        "Why use Python?",
        [{"document": "Python is useful."}],
    )

    assert "Why use Python?" in captured["prompt"]


def test_pipeline_rejects_non_string_document_text():
    pipeline = AnswerPipeline(lambda query, prompt: "Answer")

    with pytest.raises(
        TypeError,
        match="each retrieved document must contain string text",
    ):
        pipeline.run("Question?", [{"document": 123}])


def test_pipeline_rejects_non_dictionary_metadata():
    pipeline = AnswerPipeline(lambda query, prompt: "Answer")

    with pytest.raises(
        TypeError, match="document metadata must be a dictionary"
    ):
        pipeline.run(
            "Question?",
            [{"document": "Some text.", "metadata": "invalid"}],
        )
