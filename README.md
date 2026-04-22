# Enterprise RAG Assistant

## Problem
Employees struggle to find accurate answers across internal documents
(policies, manuals, runbooks). Generic chatbots hallucinate or give
uncited answers.

## Solution
A Retrieval-Augmented Generation (RAG) assistant that answers questions
**only** from provided documents and always cites its sources.

## Key Features
- Semantic search over internal documents
- Source-cited answers
- Guardrails against hallucination
- Evaluation-driven quality checks

## Tech Stack
- Python 3.12+
- FastAPI (web framework)
- ChromaDB (vector database)
- Sentence Transformers (embeddings)
- Uvicorn (ASGI server)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/rudzy123/Enterprise_Rag_Assistant.git
   cd Enterprise_Rag_Assistant
   ```

2. Create and activate virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Start the server:
   ```bash
   uvicorn main:app --reload
   ```

2. Ingest documents:
   ```bash
   curl -X POST http://localhost:8000/ingest
   ```

3. Ask questions:
   ```bash
   curl -X POST http://localhost:8000/ask \
     -H "Content-Type: application/json" \
     -d '{"question": "What is the incident response process?"}'
   ```

## API Endpoints

- `POST /ingest`: Ingest curated markdown documents from `docs/curated/`
- `POST /ask`: Answer questions using retrieved evidence

## MVP Scope (Current Stage)

This repository currently focuses on a minimal, correct RAG implementation:

- Local document ingestion
- Semantic retrieval
- Answer generation constrained to retrieved context
- Explicit refusal when evidence is insufficient

Evaluation, optimization, and UI polish are planned in later stages.

## Known Limitations

- No reranking or citation span highlighting yet
- Simple confidence heuristic
- No evaluation dataset at this stage
- No LLM integration (returns raw context)