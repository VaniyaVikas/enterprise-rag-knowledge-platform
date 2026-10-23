
import re

from src.embeddings import (
    create_embedding_model,
    embed_texts,
    embed_query,
)
from src.vector_store import VectorStore


class SelfQueryRetriever:
    """Retrieve documents using semantic similarity and metadata filters."""

    def __init__(self, documents: list[dict], model=None):
        if not isinstance(documents, list) or not documents:
            raise ValueError("documents must be a non-empty list")

        self.model = model if model is not None else create_embedding_model()
        self.documents = []
        self.metadata = []

        for item in documents:
            if not isinstance(item, dict):
                raise TypeError("each document must be a dictionary")

            document = item.get("document")
            metadata = item.get("metadata", {})

            if not isinstance(document, str) or not document.strip():
                raise ValueError("document must be a non-empty string")

            if not isinstance(metadata, dict):
                raise TypeError("metadata must be a dictionary")

            self.documents.append(document)
            self.metadata.append(metadata)

        vectors = embed_texts(self.documents, self.model)

        self.vector_store = VectorStore(dimension=len(vectors[0]))
        self.vector_store.add(
            vectors=vectors,
            documents=self.documents,
            metadata=self.metadata,
        )

    @staticmethod
    def extract_filters(query: str) -> tuple[str, dict]:
        """Split a query into semantic text and supported metadata filters."""
        if not isinstance(query, str):
            raise TypeError("query must be a string")

        if not query.strip():
            raise ValueError("query must not be empty")

        semantic_query = query
        filters = {}

        year_patterns = [
            (r"\byear\s*>\s*(\d{4})\b", ">"),
            (r"\byear\s*<\s*(\d{4})\b", "<"),
            (r"\bafter\s+(\d{4})\b", ">"),
            (r"\bbefore\s+(\d{4})\b", "<"),
        ]

        for pattern, operator in year_patterns:
            match = re.search(pattern, semantic_query, re.IGNORECASE)

            if match:
                filters["year"] = {
                    "operator": operator,
                    "value": int(match.group(1)),
                }
                semantic_query = re.sub(
                    pattern, " ", semantic_query, count=1,
                    flags=re.IGNORECASE,
                )
                break

        category_pattern = r"\bcategory\s*:\s*([a-zA-Z0-9_-]+)\b"
        category_match = re.search(
            category_pattern, semantic_query, re.IGNORECASE
        )

        if category_match:
            filters["category"] = {
                "operator": "==",
                "value": category_match.group(1).lower(),
            }
            semantic_query = re.sub(
                category_pattern, " ", semantic_query, count=1,
                flags=re.IGNORECASE,
            )

        semantic_query = re.sub(r"\s+", " ", semantic_query).strip()
        semantic_query = semantic_query.strip(" ,;:")

        if not semantic_query:
            raise ValueError("query must contain a semantic search term")

        return semantic_query, filters

    @staticmethod
    def _matches_filters(metadata: dict, filters: dict) -> bool:
        """Return whether metadata satisfies every extracted filter."""
        for field, condition in filters.items():
            if field not in metadata:
                return False

            actual_value = metadata[field]
            expected_value = condition["value"]
            operator = condition["operator"]

            if field == "year":
                try:
                    actual_value = int(actual_value)
                except (TypeError, ValueError):
                    return False

            if operator == ">" and not actual_value > expected_value:
                return False

            if operator == "<" and not actual_value < expected_value:
                return False

            if operator == "==" and (
                not isinstance(actual_value, str)
                or actual_value.lower() != expected_value
            ):
                return False

        return True

    def search(self, query: str, k: int = 3) -> list[dict]:
        """Search semantically, then apply any extracted metadata filters."""
        if not isinstance(query, str):
            raise TypeError("query must be a string")

        if not query.strip():
            raise ValueError("query must not be empty")

        if isinstance(k, bool) or not isinstance(k, int):
            raise TypeError("k must be an integer")

        if k <= 0:
            raise ValueError("k must be positive")

        semantic_query, filters = self.extract_filters(query)
        query_vector = embed_query(semantic_query, self.model)

        if filters:
            candidate_k = len(self.documents)
        else:
            candidate_k = min(k, len(self.documents))

        results = self.vector_store.search(query_vector, k=candidate_k)

        if filters:
            results = [
                result
                for result in results
                if self._matches_filters(result["metadata"] or {}, filters)
            ]

        return results[:k]
