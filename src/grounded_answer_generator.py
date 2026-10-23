
"""Generate answers and perform basic context-grounding checks."""

import re
from collections.abc import Callable


class GroundedAnswerGenerator:
    """Generate an answer and check sentence-level context overlap."""

    def __init__(
        self,
        answer_generator: Callable[[str, str], str],
        min_sentence_overlap: float = 0.3,
        no_context_answer: str = (
            "I could not find relevant information to answer this question."
        ),
    ) -> None:
        if not callable(answer_generator):
            raise TypeError("answer_generator must be callable.")

        if isinstance(min_sentence_overlap, bool) or not isinstance(
            min_sentence_overlap, (int, float)
        ):
            raise TypeError("min_sentence_overlap must be a number.")

        if not 0.0 <= min_sentence_overlap <= 1.0:
            raise ValueError(
                "min_sentence_overlap must be between 0 and 1."
            )

        if not isinstance(no_context_answer, str):
            raise TypeError("no_context_answer must be a string.")

        if not no_context_answer.strip():
            raise ValueError("no_context_answer cannot be empty.")

        self.answer_generator = answer_generator
        self.min_sentence_overlap = float(min_sentence_overlap)
        self.no_context_answer = no_context_answer.strip()

    @staticmethod
    def _tokens(text: str) -> set[str]:
        """Return lowercase word tokens from text."""
        return set(re.findall(r"\b\w+\b", text.lower()))

    @staticmethod
    def _validate_query(query: str) -> str:
        if not isinstance(query, str):
            raise TypeError("query must be a string.")

        if not query.strip():
            raise ValueError("query cannot be empty.")

        return query.strip()

    @staticmethod
    def _validate_documents(documents: list[dict]) -> list[str]:
        if not isinstance(documents, list):
            raise TypeError("retrieved_documents must be a list.")

        validated = []

        for result in documents:
            if not isinstance(result, dict):
                raise TypeError(
                    "each retrieved document must be a dictionary."
                )

            document = result.get("document")

            if not isinstance(document, str):
                raise TypeError(
                    "each retrieved document must contain string text."
                )

            if not document.strip():
                raise ValueError(
                    "retrieved document text cannot be empty."
                )

            validated.append(document.strip())

        return validated

    def check_grounding(
        self,
        answer: str,
        context: str,
    ) -> dict:
        """Estimate grounding using sentence-to-context token overlap."""
        if not isinstance(answer, str):
            raise TypeError("answer must be a string.")

        if not answer.strip():
            raise ValueError("answer cannot be empty.")

        if not isinstance(context, str):
            raise TypeError("context must be a string.")

        if not context.strip():
            raise ValueError("context cannot be empty.")

        context_tokens = self._tokens(context)
        sentences = [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+", answer.strip())
            if sentence.strip()
        ]

        sentence_results = []

        for sentence in sentences:
            sentence_tokens = self._tokens(sentence)

            if not sentence_tokens:
                overlap = 0.0
            else:
                overlap = (
                    len(sentence_tokens & context_tokens)
                    / len(sentence_tokens)
                )

            sentence_results.append(
                {
                    "sentence": sentence,
                    "overlap": overlap,
                    "grounded": overlap >= self.min_sentence_overlap,
                }
            )

        grounded_count = sum(
            item["grounded"] for item in sentence_results
        )

        total = len(sentence_results)
        score = grounded_count / total if total else 0.0

        return {
            "grounding_score": score,
            "is_grounded": total > 0 and grounded_count == total,
            "sentences": sentence_results,
        }

    def generate(
        self,
        query: str,
        retrieved_documents: list[dict],
    ) -> dict:
        """Generate an answer and return its grounding assessment."""
        query = self._validate_query(query)
        documents = self._validate_documents(retrieved_documents)

        if not documents:
            return {
                "query": query,
                "answer": self.no_context_answer,
                "sources": [],
                "grounding": {
                    "grounding_score": 0.0,
                    "is_grounded": False,
                    "sentences": [],
                },
            }

        context = "\n\n".join(
            f"[{index}] {document}"
            for index, document in enumerate(documents, start=1)
        )

        answer = self.answer_generator(query, context)

        if not isinstance(answer, str):
            raise TypeError("answer generator must return a string.")

        if not answer.strip():
            raise ValueError("generated answer cannot be empty.")

        grounding = self.check_grounding(answer.strip(), context)

        return {
            "query": query,
            "answer": answer.strip(),
            "sources": documents,
            "grounding": grounding,
        }
