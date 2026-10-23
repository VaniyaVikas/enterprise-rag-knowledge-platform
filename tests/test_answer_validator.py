
import pytest

from src.answer_validator import AnswerValidator


def test_validate_accepts_valid_answer():
    validator = AnswerValidator()

    result = validator.validate("Python is a programming language.")

    assert result == {
        "is_valid": True,
        "answer": "Python is a programming language.",
        "reason": None,
        "action": "accept",
    }


def test_validate_strips_surrounding_whitespace():
    validator = AnswerValidator()

    result = validator.validate("  A valid answer.  ")

    assert result["answer"] == "A valid answer."
    assert result["is_valid"] is True


def test_validate_rejects_empty_answer():
    validator = AnswerValidator()

    result = validator.validate("   ")

    assert result["is_valid"] is False
    assert result["reason"] == "empty_answer"
    assert result["action"] == "refuse"
    assert result["answer"] == validator.refusal_message


def test_validate_rejects_non_string_answer():
    validator = AnswerValidator()

    with pytest.raises(TypeError, match="answer must be a string"):
        validator.validate(123)


def test_validate_rejects_answer_below_minimum_length():
    validator = AnswerValidator(min_answer_length=5)

    result = validator.validate("Hey")

    assert result["is_valid"] is False
    assert result["reason"] == "answer_too_short"
    assert result["action"] == "refuse"


def test_validate_rejects_answer_above_maximum_length():
    validator = AnswerValidator(max_answer_length=5)

    result = validator.validate("This answer is too long.")

    assert result["is_valid"] is False
    assert result["reason"] == "answer_too_long"
    assert result["action"] == "refuse"


def test_validate_accepts_answer_at_minimum_length():
    validator = AnswerValidator(min_answer_length=4)

    result = validator.validate("Good")

    assert result["is_valid"] is True


def test_validate_accepts_answer_at_maximum_length():
    validator = AnswerValidator(max_answer_length=4)

    result = validator.validate("Good")

    assert result["is_valid"] is True


def test_handle_no_context_returns_fallback():
    validator = AnswerValidator()

    result = validator.handle_no_context()

    assert result["is_valid"] is True
    assert result["answer"] == validator.no_context_answer
    assert result["reason"] == "no_context"
    assert result["action"] == "fallback"


def test_handle_no_context_uses_custom_fallback():
    validator = AnswerValidator(
        no_context_answer="No relevant documents found."
    )

    result = validator.handle_no_context()

    assert result["answer"] == "No relevant documents found."


def test_validate_grounding_accepts_grounded_answer():
    validator = AnswerValidator()

    result = validator.validate_grounding(
        {"is_grounded": True, "grounding_score": 1.0}
    )

    assert result == {
        "is_valid": True,
        "reason": None,
        "action": "accept",
    }


def test_validate_grounding_refuses_unsupported_answer():
    validator = AnswerValidator(
        refusal_message="The context does not support this answer."
    )

    result = validator.validate_grounding(
        {"is_grounded": False, "grounding_score": 0.0}
    )

    assert result["is_valid"] is False
    assert result["reason"] == "unsupported_answer"
    assert result["action"] == "refuse"
    assert result["answer"] == (
        "The context does not support this answer."
    )


def test_validate_grounding_rejects_non_dictionary():
    validator = AnswerValidator()

    with pytest.raises(TypeError, match="grounding must be a dictionary"):
        validator.validate_grounding("invalid")


def test_validate_grounding_requires_boolean_status():
    validator = AnswerValidator()

    with pytest.raises(
        TypeError,
        match="grounding must contain a boolean is_grounded",
    ):
        validator.validate_grounding({"is_grounded": "yes"})


def test_constructor_rejects_non_string_fallback():
    with pytest.raises(
        TypeError,
        match="no_context_answer must be a string",
    ):
        AnswerValidator(no_context_answer=None)


def test_constructor_rejects_empty_fallback():
    with pytest.raises(
        ValueError,
        match="no_context_answer cannot be empty",
    ):
        AnswerValidator(no_context_answer="  ")


def test_constructor_rejects_non_string_refusal_message():
    with pytest.raises(
        TypeError,
        match="refusal_message must be a string",
    ):
        AnswerValidator(refusal_message=None)


def test_constructor_rejects_empty_refusal_message():
    with pytest.raises(
        ValueError,
        match="refusal_message cannot be empty",
    ):
        AnswerValidator(refusal_message="  ")


@pytest.mark.parametrize("value", [True, 1.5, "2", None])
def test_constructor_rejects_invalid_minimum_length(value):
    with pytest.raises(
        TypeError,
        match="min_answer_length must be an integer",
    ):
        AnswerValidator(min_answer_length=value)


@pytest.mark.parametrize("value", [0, -1])
def test_constructor_rejects_non_positive_minimum_length(value):
    with pytest.raises(
        ValueError,
        match="min_answer_length must be positive",
    ):
        AnswerValidator(min_answer_length=value)


@pytest.mark.parametrize("value", [True, 1.5, "10"])
def test_constructor_rejects_invalid_maximum_length(value):
    with pytest.raises(
        TypeError,
        match="max_answer_length must be an integer or None",
    ):
        AnswerValidator(max_answer_length=value)


def test_constructor_rejects_maximum_below_minimum():
    with pytest.raises(
        ValueError,
        match="max_answer_length must be greater than or equal",
    ):
        AnswerValidator(
            min_answer_length=10,
            max_answer_length=5,
        )


def test_constructor_accepts_equal_minimum_and_maximum():
    validator = AnswerValidator(
        min_answer_length=5,
        max_answer_length=5,
    )

    result = validator.validate("Hello")

    assert result["is_valid"] is True


def test_constructor_accepts_no_maximum():
    validator = AnswerValidator(max_answer_length=None)

    result = validator.validate("A longer answer is allowed.")

    assert result["is_valid"] is True


def test_validate_grounding_rejects_missing_status():
    validator = AnswerValidator()

    with pytest.raises(
        TypeError,
        match="grounding must contain a boolean is_grounded",
    ):
        validator.validate_grounding({"grounding_score": 1.0})
