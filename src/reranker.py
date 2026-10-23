
import numpy as np


class Reranker:
    """Reorder retrieved documents using a cross-encoder model."""

    DEFAULT_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(
        self,
        model=None,
        model_name: str = DEFAULT_MODEL_NAME,
    ) -> None:
        if model is not None and not callable(
            getattr(model, "predict", None)
        ):
            raise TypeError("model must provide a callable predict method.")

        if not isinstance(model_name, str):
            raise TypeError("model_name must be a string.")

        if not model_name.strip():
            raise ValueError("model_name cannot be empty.")

        self.model = model
        self.model_name = model_name

    def _load_model(self):
        """Load the cross-encoder only when it is first needed."""
        if self.model is None:
            from sentence_transformers import CrossEncoder

            self.model = CrossEncoder(self.model_name)

        return self.model

    def rerank(
        self,
        query: str,
        documents: list[dict],
        top_n: int | None = None,
    ) -> list[dict]:
        """Sort retrieved documents by query-document relevance."""
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

        model = self._load_model()

        pairs = [
            (query, item["document"])
            for item in documents
        ]

        scores = np.asarray(model.predict(pairs)).reshape(-1)

        if len(scores) != len(documents):
            raise ValueError(
                "The model must return exactly one score per document."
            )

        reranked = []

        for item, score in zip(documents, scores):
            result = item.copy()
            result["rerank_score"] = float(score)
            reranked.append(result)

        reranked.sort(
            key=lambda item: item["rerank_score"],
            reverse=True,
        )

        if top_n is not None:
            reranked = reranked[:top_n]

        return reranked
