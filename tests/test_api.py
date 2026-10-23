
"""Tests for the FastAPI application."""

import pytest
from fastapi.testclient import TestClient

from src.api import QueryRequest, create_app


class FakeRAGPipeline:
    """Provide deterministic results without loading an actual LLM."""

    def __init__(self, result=None, error=None):
        self.result = result or {
            "query": "What is RAG?",
            "answer": "RAG combines retrieval with generation.",
            "is_valid": True,
            "citations": [],
            "retrieval": {
                "requested_k": 3,
                "document_count": 1,
            },
        }
        self.error = error
        self.queries = []

    def run(self, query: str) -> dict:
        """Record the query and return a result or raise a configured error."""
        self.queries.append(query)

        if self.error is not None:
            raise self.error

        return self.result


def test_create_app_without_pipeline():
    app = create_app()

    assert app.title == "Enterprise RAG Knowledge Platform"
    assert app.state.rag_pipeline is None


def test_create_app_with_pipeline():
    pipeline = FakeRAGPipeline()
    app = create_app(pipeline)

    assert app.state.rag_pipeline is pipeline


def test_create_app_rejects_pipeline_without_run_method():
    with pytest.raises(
        TypeError,
        match="pipeline must provide a callable run method.",
    ):
        create_app(pipeline=object())


def test_query_request_strips_whitespace():
    request = QueryRequest(query="  What is RAG?  ")

    assert request.query == "What is RAG?"


@pytest.mark.parametrize("query", ["", " ", "\t", "\n", "   \t  "])
def test_query_request_rejects_blank_query(query):
    with pytest.raises(ValueError, match="query cannot be empty."):
        QueryRequest(query=query)


def test_health_endpoint_without_pipeline():
    client = TestClient(create_app())

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "pipeline_configured": False,
    }


def test_health_endpoint_with_pipeline():
    client = TestClient(create_app(FakeRAGPipeline()))

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "pipeline_configured": True,
    }


def test_query_endpoint_returns_pipeline_result():
    pipeline = FakeRAGPipeline()
    client = TestClient(create_app(pipeline))

    response = client.post(
        "/query",
        json={"query": "What is RAG?"},
    )

    assert response.status_code == 200
    assert response.json() == pipeline.result
    assert pipeline.queries == ["What is RAG?"]


def test_query_endpoint_strips_query_whitespace():
    pipeline = FakeRAGPipeline()
    client = TestClient(create_app(pipeline))

    response = client.post(
        "/query",
        json={"query": "  What is RAG?  "},
    )

    assert response.status_code == 200
    assert pipeline.queries == ["What is RAG?"]


def test_query_endpoint_without_pipeline_returns_503():
    client = TestClient(create_app())

    response = client.post(
        "/query",
        json={"query": "What is RAG?"},
    )

    assert response.status_code == 503
    assert response.json() == {
        "detail": "RAG pipeline is not configured."
    }


@pytest.mark.parametrize("query", ["", " ", "\t", "\n"])
def test_query_endpoint_rejects_blank_query(query):
    client = TestClient(create_app(FakeRAGPipeline()))

    response = client.post(
        "/query",
        json={"query": query},
    )

    assert response.status_code == 422


def test_query_endpoint_rejects_non_string_query():
    client = TestClient(create_app(FakeRAGPipeline()))

    response = client.post(
        "/query",
        json={"query": 123},
    )

    assert response.status_code == 422


def test_query_endpoint_rejects_missing_query():
    client = TestClient(create_app(FakeRAGPipeline()))

    response = client.post("/query", json={})

    assert response.status_code == 422


def test_query_endpoint_rejects_extra_fields():
    client = TestClient(create_app(FakeRAGPipeline()))

    response = client.post(
        "/query",
        json={
            "query": "What is RAG?",
            "unexpected": True,
        },
    )

    assert response.status_code == 422


def test_query_endpoint_returns_500_when_pipeline_raises():
    pipeline = FakeRAGPipeline(error=RuntimeError("internal detail"))
    client = TestClient(create_app(pipeline))

    response = client.post(
        "/query",
        json={"query": "What is RAG?"},
    )

    assert response.status_code == 500
    assert response.json() == {
        "detail": "RAG query processing failed."
    }
    assert "internal detail" not in response.text


def test_query_endpoint_rejects_non_dictionary_result():
    pipeline = FakeRAGPipeline(result=None)
    pipeline.run = lambda query: ["not", "a", "dictionary"]
    client = TestClient(create_app(pipeline))

    response = client.post(
        "/query",
        json={"query": "What is RAG?"},
    )

    assert response.status_code == 500
    assert response.json() == {
        "detail": "RAG query processing failed."
    }
