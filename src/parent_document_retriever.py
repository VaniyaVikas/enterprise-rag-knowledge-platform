from src.embeddings import create_embedding_model, embed_texts, embed_query
from src.vector_store import VectorStore


class ParentDocumentRetriever:
    """Retrieve parent documents using searchable child chunks."""

    def __init__(
        self,
        embedding_model=None,
    ) -> None:
        self.embedding_model = embedding_model
        self.vector_store = None
        self.parent_documents: dict[str, dict] = {}

    def add_documents(
        self,
        parent_documents: list[dict],
        child_chunks: list[dict],
    ) -> None:
        """Index child chunks while storing their parent documents."""

        if not isinstance(parent_documents, list):
            raise TypeError(
                "parent_documents must be a list."
            )

        if not isinstance(child_chunks, list):
            raise TypeError(
                "child_chunks must be a list."
            )

        if not parent_documents:
            raise ValueError(
                "parent_documents cannot be empty."
            )

        if not child_chunks:
            raise ValueError(
                "child_chunks cannot be empty."
            )

        self.parent_documents = {}

        for index, parent in enumerate(parent_documents):
            if not isinstance(parent, dict):
                raise TypeError(
                    f"parent_documents[{index}] must be a dictionary."
                )

            parent_id = parent.get("parent_id")
            document = parent.get("document")

            if not isinstance(parent_id, str):
                raise TypeError(
                    f"parent_documents[{index}]['parent_id'] must be a string."
                )

            if not parent_id.strip():
                raise ValueError(
                    f"parent_documents[{index}]['parent_id'] cannot be empty."
                )

            if not isinstance(document, str):
                raise TypeError(
                    f"parent_documents[{index}]['document'] must be a string."
                )

            if not document.strip():
                raise ValueError(
                    f"parent_documents[{index}]['document'] cannot be empty."
                )

            self.parent_documents[parent_id] = parent.copy()

        child_documents = []
        child_metadata = []

        for index, child in enumerate(child_chunks):
            if not isinstance(child, dict):
                raise TypeError(
                    f"child_chunks[{index}] must be a dictionary."
                )

            child_text = child.get("document")
            parent_id = child.get("parent_id")

            if not isinstance(child_text, str):
                raise TypeError(
                    f"child_chunks[{index}]['document'] must be a string."
                )

            if not child_text.strip():
                raise ValueError(
                    f"child_chunks[{index}]['document'] cannot be empty."
                )

            if not isinstance(parent_id, str):
                raise TypeError(
                    f"child_chunks[{index}]['parent_id'] must be a string."
                )

            if parent_id not in self.parent_documents:
                raise ValueError(
                    f"Unknown parent_id: {parent_id}"
                )

            child_documents.append(child_text)

            child_metadata.append(
                {
                    "parent_id": parent_id,
                    "child_index": index,
                }
            )

        model = self.embedding_model

        if model is None:
            model = create_embedding_model()
            self.embedding_model = model

        vectors = embed_texts(
            child_documents,
            model,
        )

        dimension = len(vectors[0])

        self.vector_store = VectorStore(
            dimension=dimension,
        )

        self.vector_store.add(
            vectors=vectors,
            documents=child_documents,
            metadata=child_metadata,
        )

    def search(
        self,
        query: str,
        k: int = 3,
    ) -> list[dict]:
        """Retrieve parent documents using relevant child chunks."""

        if not isinstance(query, str):
            raise TypeError(
                "query must be a string."
            )

        if not query.strip():
            raise ValueError(
                "query cannot be empty."
            )

        if isinstance(k, bool) or not isinstance(k, int):
            raise TypeError(
                "k must be an integer."
            )

        if k <= 0:
            raise ValueError(
                "k must be greater than zero."
            )

        if self.vector_store is None:
            raise ValueError(
                "No documents have been indexed."
            )

        model = self.embedding_model

        query_vector = embed_query(
            query,
            model,
        )

        child_results = self.vector_store.search(
            query_vector=query_vector,
            k=k,
        )

        results = []
        seen_parent_ids = set()

        for child_result in child_results:
            parent_id = child_result["metadata"]["parent_id"]

            if parent_id in seen_parent_ids:
                continue

            parent = self.parent_documents[parent_id]

            result = {
                "parent_id": parent_id,
                "document": parent["document"],
                "metadata": parent.get("metadata", {}),
                "score": child_result["score"],
                "child_index": child_result["metadata"]["child_index"],
            }

            results.append(result)
            seen_parent_ids.add(parent_id)

        return results