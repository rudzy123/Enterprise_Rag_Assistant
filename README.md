# Enterprise RAG Assistant

## Overview

A secure, enterprise-style Retrieval-Augmented Generation (RAG) assistant that:

- Ingests curated security documentation
- Performs semantic search with local embeddings
- Generates grounded, source-cited answers
- Requires no hardcoded credentials
- Accepts user-supplied API keys at runtime
- Degrades gracefully when LLM access is unavailable

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

## Architecture Overview

```
User Query → Retrieval → Answer Generation → Response
     ↓          ↓              ↓
  Embed Query → Semantic Search → LLM + Context → Citations
     ↓          ↓              ↓
ChromaDB ← Vector Database ← Grounded Answers ← Source Attribution
```

**Data Flow:**
1. **Ingestion**: Markdown documents → Section splitting → Embedding → ChromaDB storage
2. **Query**: User question → Embedding → Vector search → Top-k retrieval
3. **Generation**: Retrieved context + Query → LLM → Grounded answer + Citations
4. **Response**: Answer + Sources → User (degrades to retrieval-only if no API key)

## How to Run Locally

### Prerequisites
- Python 3.12+
- OpenAI API key (optional, for answer generation)

### Quick Start

1. **Setup environment:**
   ```bash
   git clone https://github.com/rudzy123/Enterprise_Rag_Assistant.git
   cd Enterprise_Rag_Assistant
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Ingest documents:**
   ```bash
   python ingestion/embed_and_store.py
   ```

3. **Launch Streamlit UI:**
   ```bash
   streamlit run app/app.py
   ```

4. **Open browser** to `http://localhost:8501`

### Alternative: Command Line

**Retrieval only:**
```bash
python retrieval/retrieve_chunks.py
```

**Answer generation (requires API key):**
```bash
OPENAI_API_KEY="your-key-here" python answer_generation/generate_answer.py
```

## How API Keys Are Handled

### Security Design
- **No hardcoded credentials**: API keys are never stored in code or configuration files
- **Runtime only**: Keys are accepted at application startup or via UI input
- **Memory-only**: Keys exist only in RAM during session, never written to disk
- **No persistence**: Keys are forgotten when application closes

### Usage Patterns

**Streamlit UI:**
- Enter API key in sidebar text field (password-masked)
- Key used for current session only
- Clear visual feedback when provided

**Command Line:**
```bash
OPENAI_API_KEY="sk-..." python answer_generation/generate_answer.py
```

**Graceful Degradation:**
- Application works without API key (retrieval-only mode)
- Clear messaging about disabled features
- No errors or crashes when LLM unavailable

## Known Limitations

### Current Implementation
- **Simple retrieval**: No reranking or hybrid search
- **Basic embeddings**: Single model (all-MiniLM-L6-v2), no fine-tuning
- **Limited context**: Fixed top-k retrieval (k=4-5)
- **No evaluation**: No quantitative metrics or test suite

### LLM Integration
- **API dependency**: Requires OpenAI API access and credits
- **Rate limits**: Subject to OpenAI API constraints
- **Cost**: Per-token pricing for answer generation
- **No offline mode**: Cannot function without internet/API access

### Enterprise Features
- **No authentication**: No user management or access controls
- **Single user**: Not designed for concurrent usage
- **No audit logging**: No tracking of queries or responses
- **No data encryption**: Documents stored in plain text

### Future Roadmap
- Multi-model support (local LLMs, other providers)
- Advanced retrieval (reranking, hybrid search)
- User management and audit logging
- Performance optimization and caching