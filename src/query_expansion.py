
import re


DEFAULT_SYNONYMS = {
    "llm": ["large language model"],
    "rag": ["retrieval augmented generation"],
    "vector database": ["vector store"],
    "embedding": ["vector representation"],
    "cheap": ["affordable", "low cost"],
}


class QueryExpander:
    """Expand search queries using configurable synonyms."""

    def __init__(
        self,
        synonym_map: dict[str, list[str]] | None = None,
    ) -> None:
        if synonym_map is None:
            synonym_map = DEFAULT_SYNONYMS

        if not isinstance(synonym_map, dict):
            raise TypeError("synonym_map must be a dictionary.")

        normalized_map = {}

        for term, synonyms in synonym_map.items():
            if not isinstance(term, str):
                raise TypeError("Each synonym key must be a string.")

            if not term.strip():
                raise ValueError("Synonym keys cannot be empty.")

            if not isinstance(synonyms, list):
                raise TypeError(
                    f"Synonyms for '{term}' must be a list."
                )

            for synonym in synonyms:
                if not isinstance(synonym, str):
                    raise TypeError("Each synonym must be a string.")

                if not synonym.strip():
                    raise ValueError("Synonyms cannot be empty.")

            normalized_map[term.strip().casefold()] = [
                synonym.strip()
                for synonym in synonyms
            ]

        self.synonym_map = normalized_map

    def expand(
        self,
        query: str,
        max_expansions: int = 5,
    ) -> list[str]:
        """Return the original query and synonym-expanded variations."""
        if not isinstance(query, str):
            raise TypeError("query must be a string.")

        if not query.strip():
            raise ValueError("query cannot be empty.")

        if (
            isinstance(max_expansions, bool)
            or not isinstance(max_expansions, int)
        ):
            raise TypeError("max_expansions must be an integer.")

        if max_expansions <= 0:
            raise ValueError(
                "max_expansions must be greater than zero."
            )

        original = query.strip()
        expanded_queries = [original]
        seen = {original.casefold()}

        for term, synonyms in self.synonym_map.items():
            pattern = re.compile(
                rf"(?<!\w){re.escape(term)}(?!\w)",
                re.IGNORECASE,
            )

            if not pattern.search(original):
                continue

            for synonym in synonyms:
                expanded = pattern.sub(
                    lambda _: synonym,
                    original,
                    count=1,
                )

                normalized = expanded.casefold()

                if normalized not in seen:
                    seen.add(normalized)
                    expanded_queries.append(expanded)

                if len(expanded_queries) >= max_expansions:
                    return expanded_queries

        return expanded_queries
