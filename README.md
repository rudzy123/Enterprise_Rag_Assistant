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
- Python
- FastAPI
- Streamlit
- Chroma (vector DB)
- LLM API (OpenAI or Anthropic)

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