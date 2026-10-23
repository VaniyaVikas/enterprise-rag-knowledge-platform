
"""Orchestrate context preparation, answer generation, validation, and citations."""

from collections.abc import Callable

from src.answer_validator import AnswerValidator
from src.context_builder import ContextBuilder
from src.grounded_answer_generator import GroundedAnswerGenerator
from src.prompt_templates import build_rag_prompt
from src.source_attribution import SourceAttributor


class AnswerPipeline:
    """Run the main answer-generation workflow using reusable components."""

    def __init__(
        self,
        answer_generator: Callable[[str, str], str],
        context_builder: ContextBuilder | None = None,
        answer_validator: AnswerValidator | None = None,
        source_attributor: SourceAttributor | None = None,
        grounding_checker: GroundedAnswerGenerator | None = None,
    ) -> None:
        if not callable(answer_generator):
            raise TypeError("answer_generator must be callable.")

        self.answer_generator = answer_generator

        self.context_builder = (
            context_builder
            if context_builder is not None
            else ContextBuilder()
        )

        self.answer_validator = (
            answer_validator
            if answer_validator is not None
            else AnswerValidator()
        )

        self.source_attributor = (
            source_attributor
            if source_attributor is not None
            else SourceAttributor()
        )

        self.grounding_checker = (
            grounding_checker
            if grounding_checker is not None
            else GroundedAnswerGenerator(answer_generator)
        )

    @staticmethod
    def _validate_query(query: str) -> str:
        if not isinstance(query, str):
            raise TypeError("query must be a string.")

        if not query.strip():
            raise ValueError("query cannot be empty.")

        return query.strip()

    @staticmethod
    def _validate_documents(documents: list[dict]) -> list[dict]:
        if not isinstance(documents, list):
            raise TypeError("retrieved_documents must be a list.")

        validated = []

        for document in documents:
            if not isinstance(document, dict):
                raise TypeError(
                    "each retrieved document must be a dictionary."
                )

            text = document.get("document")

            if not isinstance(text, str):
                raise TypeError(
                    "each retrieved document must contain string text."
                )

            if not text.strip():
                raise ValueError(
                    "retrieved document text cannot be empty."
                )

            metadata = document.get("metadata", {})

            if not isinstance(metadata, dict):
                raise TypeError("document metadata must be a dictionary.")

            validated.append(
                {
                    "document": text.strip(),
                    "metadata": metadata.copy(),
                    "index": document.get("index"),
                }
            )

        return validated

    def _fallback_result(self, query: str) -> dict:
        """Build a consistent fallback response."""
        fallback = self.answer_validator.handle_no_context()

        return {
            "query": query,
            "answer": fallback["answer"],
            "is_valid": fallback["is_valid"],
            "reason": fallback["reason"],
            "action": fallback["action"],
            "grounding": {
                "grounding_score": 0.0,
                "is_grounded": False,
                "sentences": [],
            },
            "citations": [],
        }

    def run(
        self,
        query: str,
        retrieved_documents: list[dict],
    ) -> dict:
        """Generate, validate, ground, and cite an answer."""
        query = self._validate_query(query)
        documents = self._validate_documents(retrieved_documents)

        if not documents:
            return self._fallback_result(query)

        context = self.context_builder.build(documents)

        if not context.strip():
            return self._fallback_result(query)

        prompt = build_rag_prompt(
            query=query,
            context=context,
        )

        generated_answer = self.answer_generator(query, prompt)

        validated = self.answer_validator.validate(generated_answer)

        if not validated["is_valid"]:
            return {
                "query": query,
                "answer": validated["answer"],
                "is_valid": False,
                "reason": validated["reason"],
                "action": validated["action"],
                "grounding": {
                    "grounding_score": 0.0,
                    "is_grounded": False,
                    "sentences": [],
                },
                "citations": [],
            }

        grounding = self.grounding_checker.check_grounding(
            validated["answer"],
            context,
        )

        grounding_validation = self.answer_validator.validate_grounding(
            grounding
        )

        if not grounding_validation["is_valid"]:
            return {
                "query": query,
                "answer": grounding_validation["answer"],
                "is_valid": False,
                "reason": grounding_validation["reason"],
                "action": grounding_validation["action"],
                "grounding": grounding,
                "citations": [],
            }

        attribution = self.source_attributor.attach_citations(
            validated["answer"],
            documents,
        )

        return {
            "query": query,
            "answer": attribution["answer"],
            "is_valid": True,
            "reason": None,
            "action": "accept",
            "grounding": grounding,
            "citations": attribution["citations"],
        }
