# Enterprise RAG Assistant

A production-style Retrieval-Augmented Generation (RAG) system built for enterprise security document retrieval and evaluation.

This project is engineered to prioritize retrieval quality over free-form generation: it uses pure vector search, measurable evaluation metrics, and structured debugging to improve real-world enterprise knowledge access.

## Project Overview

Enterprise teams use this system to query security documentation such as:
- NIST SP 800-53
- NIST SP 800-61
- Internal access control and incident response policies

The system was built to solve a practical problem: enterprise documentation is fragmented, and retrieval quality must be measured and improved before any generative layer is trusted.

## Key Features

- Semantic search over enterprise security content
- ChromaDB vector indexing for persistent retrieval
- Chunk-based retrieval with source metadata
- Evaluation framework for precision, recall, and hit rate
- Retrieval-only mode with no API key required
- Observability and trace debugging for retrieval failures

## Architecture

```
ingestion → embedding → vector DB → retrieval → evaluation
```

- **Ingestion**: Markdown documents are parsed, chunked, and embedded
- **Embedding**: SentenceTransformers convert text into vector representations
- **Vector DB**: ChromaDB stores document vectors and metadata
- **Retrieval**: Query embeddings search top-k nearest chunks
- **Evaluation**: Metrics drive tuning and monitor retrieval quality

## Tech Stack

- Python
- ChromaDB
- SentenceTransformers
- FastAPI
- SQLite (observability/tracing)

## Evaluation Framework

Retrieval quality is the core measure in this system. The evaluation framework computes:

- **retrieval_hit**: Whether an expected source appears in returned results
- **retrieval_precision**: Fraction of returned sources that are relevant
- **retrieval_recall**: Fraction of expected sources that were retrieved
- **missing_critical**: Expected documents not returned by retrieval
- **irrelevant_chunks**: Retrieved sources not aligned with expected targets

These metrics enable targeted improvements and expose retrieval tradeoffs clearly.

## Retrieval Optimization (Sprint 3)

This project uses evaluation signals to tune retrieval behavior:

- Top-k was increased to 8 to improve recall without losing focus
- Retrieval expands candidate results before filtering to preserve coverage
- A similarity threshold (default `0.3`) removes low-confidence chunks
- Evaluation feedback balances precision and recall for real data

The result is a retrieval pipeline that favors strong document coverage while reducing low-similarity noise.

## Performance Results

From the latest evaluation on 65 test questions:

- **Retrieval Hit Rate**: 0.85
- **Average Precision**: 0.55
- **Average Recall**: 0.85

**Interpretation**:
- Strong recall indicates good document coverage
- Moderate precision reveals semantic over-retrieval noise
- This makes the system reliable for finding relevant enterprise context while still highlighting areas for tuning

## How to Run

### Install dependencies

```bash
git clone https://github.com/rudzy123/Enterprise_Rag_Assistant.git
cd Enterprise_Rag_Assistant
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Ingest documents

```bash
python ingestion/embed_and_store.py
```

### Run retrieval

```bash
python retrieval/retrieve_chunks.py
```

### Run evaluation

```bash
python -m evals.run_evals
```

### Run API server

```bash
python main.py
```

## Project Structure

- `retrieval/` — vector search and retrieval logic
- `evals/` — evaluation harness, metrics, and test questions
- `ingestion/` — document processing, embedding, and storage pipeline
- `data/` — enterprise security documents and curated sources
- `app/` — application code and UI integration

## Future Improvements

- Add reranking models to improve precision
- Implement hybrid keyword + vector search
- Layer LLM generation on top of retrieval with grounded citations
- Add adaptive monitoring for retrieval drift and document coverage

## Conclusion

This repository demonstrates a practical, evaluation-driven RAG architecture for enterprise knowledge retrieval. It emphasizes measurable retrieval quality, data-driven tuning, and production-ready observability rather than relying purely on generative output.
