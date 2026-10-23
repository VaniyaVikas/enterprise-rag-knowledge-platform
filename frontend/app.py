
"""Flask web frontend for the Enterprise RAG Knowledge Platform."""

import os

import requests
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

API_BASE_URL = os.getenv(
    "RAG_API_URL",
    "http://127.0.0.1:8000",
)
REQUEST_TIMEOUT = 90


@app.get("/")
def home():
    """Render the chat interface."""
    return render_template("index.html")


@app.get("/health")
def health():
    """Check whether the FastAPI backend is reachable."""
    try:
        response = requests.get(
            f"{API_BASE_URL}/health",
            timeout=5,
        )
        response.raise_for_status()
        return jsonify(
            {"frontend": "ok", "backend": response.json()}
        )
    except requests.RequestException:
        return jsonify(
            {
                "frontend": "ok",
                "backend": "unavailable",
                "message": "Could not connect to the RAG API.",
            }
        ), 503


@app.post("/chat")
def chat():
    """Forward a user's question to the FastAPI RAG endpoint."""
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({"error": "A JSON request body is required."}), 400

    query = data.get("query")

    if not isinstance(query, str) or not query.strip():
        return jsonify({"error": "Please enter a non-empty question."}), 400

    try:
        response = requests.post(
            f"{API_BASE_URL}/query",
            json={"query": query.strip()},
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
    except requests.Timeout:
        return jsonify(
            {"error": "The request timed out. Please try again."}
        ), 504
    except requests.RequestException:
        return jsonify(
            {"error": "The RAG service could not process the request."}
        ), 502

    try:
        result = response.json()
    except ValueError:
        return jsonify(
            {"error": "The RAG service returned an invalid response."}
        ), 502

    return jsonify(result), response.status_code


if __name__ == "__main__":
    app.run(
        host=os.getenv("FRONTEND_HOST", "127.0.0.1"),
        port=int(os.getenv("FRONTEND_PORT", "5000")),
        debug=False,
    )
