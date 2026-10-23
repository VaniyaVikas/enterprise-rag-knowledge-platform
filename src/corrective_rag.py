
from collections.abc import Callable

from src.contextual_compression import ContextualCompressor


class CorrectiveRAG:
    """Evaluate retrieved documents and optionally use fallback retrieval."""

    def __init__(
        self,
        retriever,
        compressor: ContextualCompressor | None = None,
        fallback_retriever: Callable | object | None = None,
        min_relevant_documents: int = 1,
    ) -> None:
        if not callable(getattr(retriever, "search", None)) and not callable(
            retriever
        ):
            raise TypeError(
                "retriever must be callable or provide a callable search method"
            )

        if compressor is not None and not callable(
            getattr(compressor, "compress", None)
        ):
            raise TypeError("compressor must provide a callable compress method")

        if fallback_retriever is not None:
            if not callable(fallback_retriever) and not callable(
                getattr(fallback_retriever, "search", None)
            ):
                raise TypeError(
                    "fallback_retriever must be callable "
                    "or provide a callable search method"
                )

        if (
            isinstance(min_relevant_documents, bool)
            or not isinstance(min_relevant_documents, int)
        ):
            raise TypeError("min_relevant_documents must be an integer")

        if min_relevant_documents < 1:
            raise ValueError("min_relevant_documents must be at least 1")

        self.retriever = retriever
        self.compressor = compressor or ContextualCompressor()
        self.fallback_retriever = fallback_retriever
        self.min_relevant_documents = min_relevant_documents

    @staticmethod
    def _search(retriever, query: str, k: int) -> list[dict]:
        """Call either a retriever object's search method or a function."""
        search_method = getattr(retriever, "search", None)

        if callable(search_method):
            return search_method(query, k=k)

        return retriever(query)

    def run(self, query: str, k: int = 3) -> dict:
        """Retrieve, assess relevance with compression, and correct if needed."""
        if not isinstance(query, str):
            raise TypeError("query must be a string")

        if not query.strip():
            raise ValueError("query must not be empty")

        if isinstance(k, bool) or not isinstance(k, int):
            raise TypeError("k must be an integer")

        if k < 1:
            raise ValueError("k must be at least 1")

        normalized_query = query.strip()
        original_results = self._search(
            self.retriever,
            normalized_query,
            k,
        )

        compressed_results = self.compressor.compress(
            normalized_query,
            original_results,
        )

        fallback_used = False
        final_results = compressed_results

        if (
            len(compressed_results) < self.min_relevant_documents
            and self.fallback_retriever is not None
        ):
            fallback_results = self._search(
                self.fallback_retriever,
                normalized_query,
                k,
            )

            compressed_fallback_results = self.compressor.compress(
                normalized_query,
                fallback_results,
            )

            final_results = compressed_fallback_results
            fallback_used = True

        return {
            "query": normalized_query,
            "original_results": original_results,
            "relevant_results": final_results,
            "relevant_count": len(final_results),
            "fallback_used": fallback_used,
        }
