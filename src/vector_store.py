
from dataclasses import dataclass, field

import faiss
import numpy as np


@dataclass
class VectorStore:
    dimension: int
    index: faiss.Index = field(init=False)
    documents: list[str] = field(default_factory=list)
    metadata: list[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        if isinstance(self.dimension, bool) or not isinstance(self.dimension, int):
            raise TypeError("dimension must be an integer.")

        if self.dimension <= 0:
            raise ValueError("dimension must be greater than zero.")

        self.index = faiss.IndexFlatIP(self.dimension)

    def add(
        self,
        vectors: list[list[float]],
        documents: list[str],
        metadata: list[dict] | None = None,
    ) -> None:
        if not isinstance(vectors, list):
            raise TypeError("vectors must be a list.")

        if not isinstance(documents, list):
            raise TypeError("documents must be a list.")

        if len(vectors) != len(documents):
            raise ValueError("vectors and documents must have the same length.")

        if not vectors:
            return

        if not all(isinstance(document, str) and document.strip() for document in documents):
            raise ValueError("documents must contain non-empty strings.")

        if metadata is None:
            metadata = [{} for _ in documents]

        if not isinstance(metadata, list) or len(metadata) != len(documents):
            raise ValueError("metadata must match the number of documents.")

        if not all(isinstance(item, dict) for item in metadata):
            raise TypeError("each metadata item must be a dictionary.")

        try:
            matrix = np.asarray(vectors, dtype=np.float32)
        except (TypeError, ValueError) as error:
            raise ValueError("vectors must contain valid numeric values.") from error

        if matrix.shape != (len(vectors), self.dimension):
            raise ValueError("vectors must match the configured dimension.")

        if not np.isfinite(matrix).all():
            raise ValueError("vectors must contain only finite values.")

        norms = np.linalg.norm(matrix, axis=1)

        if np.any(norms == 0):
            raise ValueError("zero vectors are not supported.")

        matrix = np.ascontiguousarray(matrix / norms[:, np.newaxis])

        self.index.add(matrix)
        self.documents.extend(documents)
        self.metadata.extend(metadata)

    def search(
        self,
        query_vector: list[float],
        k: int = 3,
    ) -> list[dict]:
        if not isinstance(query_vector, list):
            raise TypeError("query_vector must be a list.")

        if isinstance(k, bool) or not isinstance(k, int):
            raise TypeError("k must be an integer.")

        if k <= 0:
            raise ValueError("k must be greater than zero.")

        if not query_vector:
            raise ValueError("query_vector cannot be empty.")

        try:
            query = np.asarray([query_vector], dtype=np.float32)
        except (TypeError, ValueError) as error:
            raise ValueError("query_vector must contain valid numeric values.") from error

        if query.shape != (1, self.dimension):
            raise ValueError("query_vector must match the configured dimension.")

        if not np.isfinite(query).all():
            raise ValueError("query_vector must contain only finite values.")

        norm = np.linalg.norm(query)

        if norm == 0:
            raise ValueError("zero query vectors are not supported.")

        query = np.ascontiguousarray(query / norm)

        if self.index.ntotal == 0:
            return []

        scores, indices = self.index.search(
            query,
            min(k, self.index.ntotal),
        )

        results = []

        for score, index_id in zip(scores[0], indices[0]):
            if index_id < 0:
                continue

            results.append(
                {
                    "document": self.documents[index_id],
                    "metadata": self.metadata[index_id],
                    "score": float(score),
                    "index": int(index_id),
                }
            )

        return results
