
import re


class ContextualCompressor:
    """Select query-relevant sentences from retrieved documents."""

    def __init__(
        self,
        min_sentence_length: int = 3,
    ) -> None:
        if (
            isinstance(min_sentence_length, bool)
            or not isinstance(min_sentence_length, int)
        ):
            raise TypeError(
                "min_sentence_length must be an integer."
            )

        if min_sentence_length <= 0:
            raise ValueError(
                "min_sentence_length must be greater than zero."
            )

        self.min_sentence_length = min_sentence_length

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        """Convert text into lowercase alphanumeric tokens."""
        return set(re.findall(r"\b\w+\b", text.lower()))

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """Split text at sentence-ending punctuation."""
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())

        return [
            sentence.strip()
            for sentence in sentences
            if sentence.strip()
        ]

    def compress(
        self,
        query: str,
        documents: list[dict],
        top_n: int | None = None,
    ) -> list[dict]:
        """Keep only documents containing query-relevant sentences."""
        if not isinstance(query, str):
            raise TypeError("query must be a string.")

        if not query.strip():
            raise ValueError("query cannot be empty.")

        if not isinstance(documents, list):
            raise TypeError("documents must be a list.")

        if top_n is not None:
            if isinstance(top_n, bool) or not isinstance(top_n, int):
                raise TypeError("top_n must be an integer or None.")

            if top_n <= 0:
                raise ValueError("top_n must be greater than zero.")

        for index, item in enumerate(documents):
            if not isinstance(item, dict):
                raise TypeError(
                    f"documents[{index}] must be a dictionary."
                )

            document = item.get("document")

            if not isinstance(document, str):
                raise TypeError(
                    f"documents[{index}]['document'] must be a string."
                )

            if not document.strip():
                raise ValueError(
                    f"documents[{index}]['document'] cannot be empty."
                )

        if not documents:
            return []

        query_tokens = self._tokenize(query)
        minimum_overlap = min(2, len(query_tokens))
        compressed_results = []

        for item in documents:
            sentences = self._split_sentences(item["document"])
            relevant_sentences = []

            for sentence in sentences:
                sentence_tokens = self._tokenize(sentence)

                if len(sentence_tokens) < self.min_sentence_length:
                    continue

                overlap = query_tokens.intersection(sentence_tokens)

                if len(overlap) >= minimum_overlap:
                    relevant_sentences.append(
                        (sentence, len(overlap))
                    )

            if not relevant_sentences:
                continue

            relevant_sentences.sort(
                key=lambda pair: pair[1],
                reverse=True,
            )

            compressed_text = " ".join(
                sentence
                for sentence, _ in relevant_sentences
            )

            result = item.copy()
            result["document"] = compressed_text
            result["compression_ratio"] = (
                len(compressed_text) / len(item["document"])
            )
            compressed_results.append(result)

        if top_n is not None:
            compressed_results = compressed_results[:top_n]

        return compressed_results
