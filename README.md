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

## Evaluation Framework

### Why Evaluate Before Answer Generation?
Evaluation is performed at the **retrieval layer** before the LLM ever sees the query. This design:
- Separates signal (quality retrieval) from noise (LLM generation artifacts)
- Identifies bottlenecks early—poor retrieval cannot be fixed by prompting
- Reduces costs by failing fast rather than paying for bad generations
- Enables rapid iteration on chunking, embedding, and ranking strategies

### JSONL Evaluation Dataset

Located in `evals/questions.jsonl`, each line contains a question object:

```json
{"id": "Q1", "question": "What is the company policy on VPN usage?", "expected_sources": ["network_policy.md"], "answerable": true}
```

**Fields:**
- `id`: Unique question identifier
- `question`: The question text
- `expected_sources`: List of document filenames that should be retrieved (empty list for answerability controls)
- `answerable`: Boolean indicating whether the question should be answerable from the corpus

### Evaluation Metrics

Run evaluations with:
```bash
python evals/run_evals.py
```

Three key metrics are computed per question:

| Metric | Definition | Target |
|--------|-----------|--------|
| **Retrieval Hit Rate** | % of questions where at least one expected source appears in top-k results | > 80% |
| **Answerability Accuracy** | % of questions where confidence ≥ 0.3 matches the `answerable` field | > 85% |
| **Citation Rate** | % of high-confidence answers that return non-empty source lists; low-confidence answers return no sources | 100% |

Example output:
```
Retrieval Hit Rate: 75.00%
Answerability Accuracy: 75.00%
Citation Rate: 75.00%
Average Confidence: 0.50
```

### Using Evaluation Results for Tuning

Evaluation guides iterative improvements:

1. **Low retrieval hit rate** → Re-examine chunking strategy, embedding quality, or top-k value
2. **Poor answerability accuracy** → Adjust confidence threshold (0.3 default) or improve metadata filtering
3. **Missing citations** → Ensure metadata (source file names) is properly propagated through the retrieval pipeline

This feedback loop ensures the system reliable before deploying to users.

### Confidence Scoring Strategy

**Key Principle**: Confidence is derived from **retrieval quality signals**, not LLM generation heuristics.

This design choice prevents the system from hallucinating high-confidence answers. If retrieval is weak, the system refuses to answer—no amount of LLM prompt engineering can fix bad source material.

**Confidence Signals:**

| Signal | Weight | Interpretation |
|--------|--------|-----------------|
| **Similarity Score** | 55% | Average L2 distance from query embedding to retrieved chunks (lower distance = higher similarity) |
| **Document Count** | 30% | Number of relevant chunks retrieved; multiple matches strengthen signal (max 3) |
| **Source Consolidation** | 15% | Whether multiple chunks come from the same source document; same-source chunks indicate concentrated support |

**Score Calculation:**
```
confidence = 0.55 × avg_similarity + 0.30 × doc_count_score + 0.15 × source_consistency
```

**Confidence Ranges:**

| Range | Score | Behavior | Use Case |
|-------|-------|----------|----------|
| **Weak** | 0.0–0.3 | System refuses to answer | Unanswerable questions, weak matches |
| **Partial** | 0.3–0.6 | Answer provided with caution | Relevant but incomplete evidence |
| **Strong** | 0.6–1.0 | High-confidence answer | Multiple strong matches, well-supported |

The **0.3 threshold** is the key guardrail: below it, the system refuses to answer. This prevents generating unreliable responses for out-of-corpus questions.

**Why Retrieval-Based Confidence?**
- **Transparency**: Score directly reflects data quality, not model uncertainty
- **Debuggability**: Confidence reasons include source counts and similarity metrics
- **Controllability**: Confidence can be tuned by adjusting chunking, embedding, or retrieval top-k
- **Trustworthiness**: Failures are predictable and attributable to source material, not hallucination

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

## License

This project is licensed under the Creative Commons
Attribution-NonCommercial-NoDerivatives 4.0 International License.

You are free to:
- View and run the software
- Use it for personal or internal purposes
- Configure it with your own API keys

You may not:
- Redistribute modified versions
- Use it for commercial purposes
- Claim authorship of the original work

© 2026 Rudolf Musika
