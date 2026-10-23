
"""FastAPI application for the production RAG knowledge platform."""

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, field_validator


class QueryRequest(BaseModel):
    """Validate incoming user queries."""

    model_config = ConfigDict(extra="forbid")

    query: str

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        """Reject blank queries and remove surrounding whitespace."""
        if not value.strip():
            raise ValueError("query cannot be empty.")

        return value.strip()


def create_app(pipeline: Any = None) -> FastAPI:
    """Create the API application with an optional injected RAG pipeline."""
    if pipeline is not None and not callable(
        getattr(pipeline, "run", None)
    ):
        raise TypeError("pipeline must provide a callable run method.")

    app = FastAPI(
        title="Enterprise RAG Knowledge Platform",
        description="API for retrieving grounded answers from a knowledge base.",
        version="1.0.0",
    )

    app.state.rag_pipeline = pipeline

    @app.get("/health")
    def health_check() -> dict:
        """Report API health and pipeline configuration status."""
        return {
            "status": "ok",
            "pipeline_configured": app.state.rag_pipeline is not None,
        }

    @app.post("/query")
    def query_knowledge_base(request: QueryRequest) -> dict:
        """Run a query through the configured RAG pipeline."""
        rag_pipeline = app.state.rag_pipeline

        if rag_pipeline is None:
            raise HTTPException(
                status_code=503,
                detail="RAG pipeline is not configured.",
            )

        try:
            result = rag_pipeline.run(request.query)

            if not isinstance(result, dict):
                raise TypeError(
                    "RAG pipeline must return a dictionary."
                )

            return result

        except HTTPException:
            raise

        except Exception as error:
            raise HTTPException(
                status_code=500,
                detail="RAG query processing failed.",
            ) from error

    return app


app = create_app()
