
import pytest

from src.grounded_answer_generator import GroundedAnswerGenerator


def fake_answer_generator(query: str, context: str) -> str:
    return "Python is a programming language."


def test_generate_returns_answer_and_grounding_report():
    generator = GroundedAnswerGenerator(fake_answer_generator)

    result = generator.generate(
        "What is Python?",
        [{"document": "Python is a programming language."}],
    )

    assert result["query"] == "What is Python?"
    assert result["answer"] == "Python is a programming language."
    assert result["grounding"]["is_grounded"] is True
    assert result["grounding"]["grounding_score"] == 1.0
    assert len(result["sources"]) == 1


def test_generate_returns_fallback_without_documents():
    generator = GroundedAnswerGenerator(fake_answer_generator)

    result = generator.generate("What is Python?", [])

    assert result["sources"] == []
    assert result["answer"] == (
        "I could not find relevant information to answer this question."
    )
    assert result["grounding"]["is_grounded"] is False


def test_generate_uses_custom_fallback():
    generator = GroundedAnswerGenerator(
        fake_answer_generator,
        no_context_answer="No supporting context.",
    )

    result = generator.generate("Question?", [])

    assert result["answer"] == "No supporting context."


def test_generator_receives_query_and_numbered_context():
    received = {}

    def capture_generator(query: str, context: str) -> str:
        received["query"] = query
        received["context"] = context
        return "Python is popular."

    generator = GroundedAnswerGenerator(capture_generator)

    generator.generate(
        "Tell me about Python",
        [{"document": "Python is popular."}],
    )

    assert received["query"] == "Tell me about Python"
    assert received["context"] == "[1] Python is popular."


def test_generate_strips_query_and_answer():
    generator = GroundedAnswerGenerator(
        lambda query, context: "  Python is useful.  "
    )

    result = generator.generate(
        "  What is Python?  ",
        [{"document": "Python is useful."}],
    )

    assert result["query"] == "What is Python?"
    assert result["answer"] == "Python is useful."


def test_check_grounding_returns_sentence_details():
    generator = GroundedAnswerGenerator(
        fake_answer_generator,
        min_sentence_overlap=0.5,
    )

    result = generator.check_grounding(
        "Python is useful. Java runs on a virtual machine.",
        "Python is useful for programming.",
    )

    assert len(result["sentences"]) == 2
    assert result["sentences"][0]["grounded"] is True
    assert result["sentences"][1]["grounded"] is False
    assert result["is_grounded"] is False


def test_check_grounding_uses_configured_threshold():
    generator = GroundedAnswerGenerator(
        fake_answer_generator,
        min_sentence_overlap=0.75,
    )

    result = generator.check_grounding(
        "Python is a programming language.",
        "Python is a programming language.",
    )

    assert result["sentences"][0]["overlap"] == 1.0
    assert result["is_grounded"] is True


def test_check_grounding_rejects_non_string_answer():
    generator = GroundedAnswerGenerator(fake_answer_generator)

    with pytest.raises(TypeError, match="answer must be a string"):
        generator.check_grounding(123, "Some context.")


def test_check_grounding_rejects_empty_answer():
    generator = GroundedAnswerGenerator(fake_answer_generator)

    with pytest.raises(ValueError, match="answer cannot be empty"):
        generator.check_grounding("  ", "Some context.")


def test_check_grounding_rejects_non_string_context():
    generator = GroundedAnswerGenerator(fake_answer_generator)

    with pytest.raises(TypeError, match="context must be a string"):
        generator.check_grounding("Some answer.", 123)


def test_check_grounding_rejects_empty_context():
    generator = GroundedAnswerGenerator(fake_answer_generator)

    with pytest.raises(ValueError, match="context cannot be empty"):
        generator.check_grounding("Some answer.", "  ")


def test_constructor_rejects_non_callable_generator():
    with pytest.raises(TypeError, match="answer_generator must be callable"):
        GroundedAnswerGenerator(None)


@pytest.mark.parametrize("threshold", [-0.1, 1.1])
def test_constructor_rejects_threshold_outside_range(threshold):
    with pytest.raises(ValueError, match="between 0 and 1"):
        GroundedAnswerGenerator(
            fake_answer_generator,
            min_sentence_overlap=threshold,
        )


@pytest.mark.parametrize("threshold", [True, "0.5", None])
def test_constructor_rejects_non_numeric_threshold(threshold):
    with pytest.raises(TypeError, match="must be a number"):
        GroundedAnswerGenerator(
            fake_answer_generator,
            min_sentence_overlap=threshold,
        )


def test_constructor_rejects_invalid_fallback_type():
    with pytest.raises(TypeError, match="no_context_answer must be a string"):
        GroundedAnswerGenerator(
            fake_answer_generator,
            no_context_answer=None,
        )


def test_constructor_rejects_empty_fallback():
    with pytest.raises(ValueError, match="no_context_answer cannot be empty"):
        GroundedAnswerGenerator(
            fake_answer_generator,
            no_context_answer="  ",
        )


def test_generate_rejects_non_string_query():
    generator = GroundedAnswerGenerator(fake_answer_generator)

    with pytest.raises(TypeError, match="query must be a string"):
        generator.generate(123, [])


def test_generate_rejects_empty_query():
    generator = GroundedAnswerGenerator(fake_answer_generator)

    with pytest.raises(ValueError, match="query cannot be empty"):
        generator.generate("  ", [])


def test_generate_rejects_non_list_documents():
    generator = GroundedAnswerGenerator(fake_answer_generator)

    with pytest.raises(TypeError, match="retrieved_documents must be a list"):
        generator.generate("Question?", "not a list")


def test_generate_rejects_non_dictionary_document():
    generator = GroundedAnswerGenerator(fake_answer_generator)

    with pytest.raises(TypeError, match="each retrieved document must be a dictionary"):
        generator.generate("Question?", ["not a dictionary"])


def test_generate_rejects_missing_document_text():
    generator = GroundedAnswerGenerator(fake_answer_generator)

    with pytest.raises(
        TypeError,
        match="each retrieved document must contain string text",
    ):
        generator.generate("Question?", [{"metadata": {}}])


def test_generate_rejects_non_string_document_text():
    generator = GroundedAnswerGenerator(fake_answer_generator)

    with pytest.raises(
        TypeError,
        match="each retrieved document must contain string text",
    ):
        generator.generate("Question?", [{"document": 123}])


def test_generate_rejects_empty_document_text():
    generator = GroundedAnswerGenerator(fake_answer_generator)

    with pytest.raises(
        ValueError,
        match="retrieved document text cannot be empty",
    ):
        generator.generate("Question?", [{"document": "  "}])


def test_generate_rejects_non_string_generated_answer():
    generator = GroundedAnswerGenerator(lambda query, context: 123)

    with pytest.raises(TypeError, match="answer generator must return a string"):
        generator.generate("Question?", [{"document": "Some context."}])


def test_generate_rejects_empty_generated_answer():
    generator = GroundedAnswerGenerator(lambda query, context: "  ")

    with pytest.raises(ValueError, match="generated answer cannot be empty"):
        generator.generate("Question?", [{"document": "Some context."}])


def test_tokens_are_case_insensitive():
    generator = GroundedAnswerGenerator(fake_answer_generator)

    assert generator._tokens("Python PYTHON python") == {"python"}


def test_generate_numbers_multiple_documents():
    received = {}

    def capture_generator(query: str, context: str) -> str:
        received["context"] = context
        return "Python is useful."

    generator = GroundedAnswerGenerator(capture_generator)

    generator.generate(
        "Question?",
        [
            {"document": "Python is useful."},
            {"document": "Python supports libraries."},
        ],
    )

    assert received["context"] == (
        "[1] Python is useful.\n\n"
        "[2] Python supports libraries."
    )
