
from rank_bm25 import BM25Okapi


class BM25Retriever:
    """Retrieve documents using BM25 keyword relevance."""

    def __init__(self, documents: list[str]) -> None:
        if not isinstance(documents, list):
            raise TypeError("documents must be a list.")

        if not documents:
            raise ValueError("documents cannot be empty.")

        if not all(
            isinstance(document, str) and document.strip()
            for document in documents
        ):
            raise ValueError("documents must contain non-empty strings.")

        self.documents = documents.copy()
        self.tokenized_documents = [
            document.lower().split()
            for document in self.documents
        ]

        self.index = BM25Okapi(self.tokenized_documents)

    def search(self, query: str, k: int = 3) -> list[dict]:
        if not isinstance(query, str):
            raise TypeError("query must be a string.")

        if not query.strip():
            raise ValueError("query cannot be empty.")

        if isinstance(k, bool) or not isinstance(k, int):
            raise TypeError("k must be an integer.")

        if k <= 0:
            raise ValueError("k must be greater than zero.")

        tokenized_query = query.lower().split()
        scores = self.index.get_scores(tokenized_query)

        ranked_indices = sorted(
            range(len(self.documents)),
            key=lambda index: scores[index],
            reverse=True,
        )

        results = []

        for index in ranked_indices[:k]:
            results.append(
                {
                    "document": self.documents[index],
                    "score": float(scores[index]),
                    "index": index,
                }
            )

        return results
