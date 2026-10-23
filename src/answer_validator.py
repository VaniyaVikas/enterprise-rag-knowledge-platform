
"""Validate generated answers and handle refusal or fallback responses."""


class AnswerValidator:
    """Validate answer text and apply configurable refusal handling."""

    def __init__(
        self,
        no_context_answer: str = (
            "I could not find relevant information to answer this question."
        ),
        refusal_message: str = (
            "I cannot provide an answer based on the available information."
        ),
        min_answer_length: int = 1,
        max_answer_length: int | None = None,
    ) -> None:
        if not isinstance(no_context_answer, str):
            raise TypeError("no_context_answer must be a string.")

        if not no_context_answer.strip():
            raise ValueError("no_context_answer cannot be empty.")

        if not isinstance(refusal_message, str):
            raise TypeError("refusal_message must be a string.")

        if not refusal_message.strip():
            raise ValueError("refusal_message cannot be empty.")

        if isinstance(min_answer_length, bool) or not isinstance(
            min_answer_length, int
        ):
            raise TypeError("min_answer_length must be an integer.")

        if min_answer_length <= 0:
            raise ValueError("min_answer_length must be positive.")

        if max_answer_length is not None:
            if isinstance(max_answer_length, bool) or not isinstance(
                max_answer_length, int
            ):
                raise TypeError(
                    "max_answer_length must be an integer or None."
                )

            if max_answer_length < min_answer_length:
                raise ValueError(
                    "max_answer_length must be greater than or equal "
                    "to min_answer_length."
                )

        self.no_context_answer = no_context_answer.strip()
        self.refusal_message = refusal_message.strip()
        self.min_answer_length = min_answer_length
        self.max_answer_length = max_answer_length

    def validate(self, answer: str) -> dict:
        """Validate answer text and return status plus normalized answer."""
        if not isinstance(answer, str):
            raise TypeError("answer must be a string.")

        normalized_answer = answer.strip()

        if not normalized_answer:
            return {
                "is_valid": False,
                "answer": self.refusal_message,
                "reason": "empty_answer",
                "action": "refuse",
            }

        if len(normalized_answer) < self.min_answer_length:
            return {
                "is_valid": False,
                "answer": self.refusal_message,
                "reason": "answer_too_short",
                "action": "refuse",
            }

        if (
            self.max_answer_length is not None
            and len(normalized_answer) > self.max_answer_length
        ):
            return {
                "is_valid": False,
                "answer": self.refusal_message,
                "reason": "answer_too_long",
                "action": "refuse",
            }

        return {
            "is_valid": True,
            "answer": normalized_answer,
            "reason": None,
            "action": "accept",
        }

    def handle_no_context(self) -> dict:
        """Return a safe fallback when no retrieved context is available."""
        return {
            "is_valid": True,
            "answer": self.no_context_answer,
            "reason": "no_context",
            "action": "fallback",
        }

    def validate_grounding(self, grounding: dict) -> dict:
        """Refuse answers when the grounding report says they are unsupported."""
        if not isinstance(grounding, dict):
            raise TypeError("grounding must be a dictionary.")

        is_grounded = grounding.get("is_grounded")

        if not isinstance(is_grounded, bool):
            raise TypeError("grounding must contain a boolean is_grounded.")

        if is_grounded:
            return {
                "is_valid": True,
                "reason": None,
                "action": "accept",
            }

        return {
            "is_valid": False,
            "reason": "unsupported_answer",
            "action": "refuse",
            "answer": self.refusal_message,
        }
