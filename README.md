# Enterprise RAG Knowledge Platform

A production-oriented Retrieval-Augmented Generation (RAG) platform built with Python, LangChain-compatible components, FAISS, Hugging Face embeddings, Groq LLMs, and FastAPI.

## Overview

This project demonstrates an end-to-end enterprise RAG architecture:

1. Document ingestion
2. Document processing
3. Configurable text chunking
4. Metadata-aware chunks
5. Local embedding generation
6. Vector similarity search
7. Multiple retrieval strategies
8. Advanced RAG techniques
9. Grounded answer generation
10. Source attribution
11. RAG evaluation
12. Production RAG pipeline
13. Groq LLM integration
14. FastAPI application
15. Automated testing

The goal is to build a strong foundation for production-grade GenAI and Agentic AI systems.

---

## Architecture

```text
Documents
   |
   v
Document Loaders
   |
   v
Document Processor
   |
   v
Chunking + Metadata
   |
   v
Hugging Face Embeddings
   |
   v
FAISS Vector Store
   |
   v
Retrieval Layer
   |
   +--> Similarity Retrieval
   +--> MMR
   +--> BM25
   +--> Hybrid Retrieval
   +--> Multi Query
   +--> Query Expansion
   +--> HyDE
   +--> Parent Document Retrieval
   +--> Self Query
   +--> Contextual Retrieval
   +--> Reranking
   |
   v
Advanced RAG
   |
   +--> Query Routing
   +--> Query Decomposition
   +--> RAG Fusion
   +--> Corrective RAG
   +--> Adaptive RAG
   |
   v
Context Builder
   |
   v
Grounded Answer Generation
   |
   v
Source Attribution
   |
   v
Production RAG Pipeline
   |
   v
FastAPI
   |
   +--> GET  /health
   +--> POST /query
```

---

## Main Features

### Document Processing

- TXT document loading
- PDF loading
- DOCX loading
- HTML loading
- Web document loading
- Configurable chunking
- Recursive text splitting
- Metadata-aware chunking

### Embeddings

The project uses:

```text
sentence-transformers/all-MiniLM-L6-v2
```

Embeddings are generated locally using Hugging Face sentence-transformers.

### Vector Store

FAISS is used for dense vector similarity search.

The current vector store uses normalized vectors with inner-product similarity.

### Retrieval

Implemented retrieval components include:

- Basic vector retrieval
- BM25 retrieval
- Hybrid retrieval
- Ensemble retrieval
- MMR retrieval
- Multi-query retrieval
- Query expansion
- HyDE retrieval
- Parent-document retrieval
- Self-query retrieval
- Contextual retrieval
- Reranking

### Advanced RAG

The project includes:

- Query routing
- Query decomposition
- RAG Fusion
- Corrective RAG
- Adaptive RAG
- Context compression
- Context building
- Grounded answer generation
- Answer validation
- Source attribution

### LLM

Groq is used for answer generation.

The configured model can be controlled through:

```text
GROQ_MODEL
```

Default:

```text
openai/gpt-oss-120b
```

### API

FastAPI exposes:

```text
GET /health
POST /query
```

Swagger documentation is available at:

```text
http://127.0.0.1:8000/docs
```

---

## Project Structure

```text
gen1/
│
├── configs/
│   └── chunking.yaml
│
├── data/
│   ├── raw/
│   └── processed/
│
├── src/
│   ├── adaptive_rag.py
│   ├── advanced_chunking.py
│   ├── answer_generator.py
│   ├── answer_pipeline.py
│   ├── answer_validator.py
│   ├── api.py
│   ├── bm25_retriever.py
│   ├── chunking.py
│   ├── config_loader.py
│   ├── contextual_compression.py
│   ├── contextual_retrieval.py
│   ├── context_builder.py
│   ├── corrective_rag.py
│   ├── document_chunk.py
│   ├── document_loader.py
│   ├── document_processor.py
│   ├── docx_loader.py
│   ├── embeddings.py
│   ├── ensemble_retriever.py
│   ├── evaluation_metrics.py
│   ├── groq_answer_generator.py
│   ├── groq_rag_pipeline.py
│   ├── grounded_answer_generator.py
│   ├── html_loader.py
│   ├── hyde_retriever.py
│   ├── hybrid_retriever.py
│   ├── llm_factory.py
│   ├── metadata_chunking.py
│   ├── mmr_retriever.py
│   ├── multi_query_retriever.py
│   ├── parent_document_retriever.py
│   ├── pdf_loader.py
│   ├── process_document.py
│   ├── production_rag_pipeline.py
│   ├── prompt_templates.py
│   ├── query_decomposition.py
│   ├── query_expansion.py
│   ├── query_router.py
│   ├── rag_fusion.py
│   ├── reranker.py
│   ├── retrieval.py
│   ├── self_query_retriever.py
│   ├── source_attribution.py
│   ├── vector_similarity.py
│   ├── vector_store.py
│   ├── web_loader.py
│   └── application.py
│
├── tests/
│
├── .env
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Requirements

- Python 3.11+
- pip
- Internet connection for initial model/package downloads
- Groq API key for Groq-powered answer generation

---

## Installation

Create and activate a virtual environment:

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

---

## Environment Configuration

Create a `.env` file in the project root.

Example:

```dotenv
GROQ_API_KEY=your-groq-api-key
GROQ_MODEL=openai/gpt-oss-120b

EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

API_HOST=127.0.0.1
API_PORT=8000

RETRIEVAL_K=3
```

Never commit `.env` to Git.

The `.env.example` file should contain placeholders only and must never contain a real API key.

---

## Chunking Configuration

Chunking can be configured through:

```text
configs/chunking.yaml
```

Example:

```yaml
chunking:
  chunk_size: 100
  overlap: 20
```

---

## Running the API

Start the application:

```powershell
uvicorn src.application:app --host 127.0.0.1 --port 8000
```

The API will be available at:

```text
http://127.0.0.1:8000
```

---

## Health Check

Open:

```text
http://127.0.0.1:8000/health
```

Expected response:

```json
{
  "status": "ok",
  "pipeline_configured": true
}
```

---

## Query API

Endpoint:

```text
POST /query
```

Request:

```json
{
  "query": "What does RAG stand for?"
}
```

The API validates the query, retrieves relevant document chunks, sends grounded context to the answer generator, and returns the resulting response.

---

## Swagger Documentation

Interactive API documentation:

```text
http://127.0.0.1:8000/docs
```

OpenAPI schema:

```text
http://127.0.0.1:8000/openapi.json
```

---

## Running Tests

Run the complete test suite:

```powershell
python -m pytest -v
```

The project currently has:

```text
1011 passed
```

This includes unit, integration, retrieval, RAG pipeline, Groq integration, and API tests.

---

## Testing Individual Components

Example:

```powershell
python -m pytest tests/test_api.py -v
```

```powershell
python -m pytest tests/test_retrieval.py -v
```

```powershell
python -m pytest tests/test_groq_rag_pipeline.py -v
```

---

## Security

Secrets must never be committed to source control.

Important files:

```text
.env
.env.example
.gitignore
```

`.env` contains the real API key.

`.env.example` contains only placeholders.

If an API key is accidentally exposed, revoke it immediately and create a new key.

---

## Production Considerations

The current project provides a strong production-oriented architecture, but additional infrastructure would be required for a large-scale deployment.

Possible future improvements:

- Persistent vector database
- PostgreSQL metadata storage
- Redis caching
- Background ingestion workers
- Authentication and authorization
- Rate limiting
- Observability
- Distributed tracing
- Structured logging
- Docker
- CI/CD
- Cloud deployment
- Horizontal scaling
- Persistent document storage
- Production-grade reranking models

---

## Current Status

### Enterprise RAG Knowledge Platform

**Status: Feature-complete prototype / production-oriented foundation**

Implemented:

- Document ingestion
- Configurable chunking
- Embeddings
- FAISS vector search
- Advanced retrieval
- Advanced RAG
- Answer generation
- Grounding
- Source attribution
- Evaluation
- Production pipeline
- Groq integration
- FastAPI API
- Automated test suite

Test status:

```text
1011 passed
```

---

## Learning Objectives Covered

This project provides practical experience with:

- Python AI application architecture
- LangChain concepts
- RAG architecture
- LCEL-style pipeline design
- Embeddings
- Vector databases
- Semantic search
- Hybrid retrieval
- Query transformation
- Reranking
- Context management
- Grounded generation
- Evaluation
- FastAPI
- LLM integration
- Production-oriented design
- Automated testing

---

## Next Project

This project is the first major project in the larger Agentic AI / GenAI engineering roadmap.

The next project will build on these foundations and move deeper into:

- LangGraph
- Stateful workflows
- Agent orchestration
- Tool calling
- Memory
- Multi-agent systems
- Human-in-the-loop workflows
- Production agent architecture