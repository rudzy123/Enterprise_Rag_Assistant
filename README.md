# Enterprise RAG Assistant

## Overview

A secure, enterprise-style Retrieval-Augmented Generation (RAG) assistant that:

- Ingests curated security documentation
- Performs semantic search with local embeddings
- Generates grounded, source-cited answers
- Requires no hardcoded credentials
- Accepts user-supplied API keys at runtime
- Degrades gracefully when LLM access is unavailable

## Business Problem

Enterprise knowledge management faces critical challenges:
- **Information Fragmentation**: Critical policies and procedures scattered across documents
- **Access Barriers**: Employees struggle to locate accurate answers from internal documentation
- **Quality Risks**: Generic AI chatbots frequently hallucinate or provide uncited, unreliable answers
- **Compliance Concerns**: Incorrect information can lead to security violations or operational errors
- **Scalability Issues**: Manual knowledge curation doesn't scale with document growth

## Why RAG Systems Fail in Production

Traditional RAG implementations suffer from:
- **Retrieval Gaps**: Poor embedding quality or indexing misses critical context
- **Generation Hallucinations**: LLMs fabricate information despite retrieval context
- **Quality Blindness**: No systematic evaluation of answer accuracy and grounding
- **Maintenance Burden**: No automated monitoring of system performance degradation
- **Debugging Opacity**: Difficult to diagnose why specific answers fail

## How This System Reduces Hallucination Risk

This evaluation-driven RAG system implements multiple safeguards:

- **Groundedness Evaluation**: Every answer scored for factual support (70%+ threshold)
- **Importance-Weighted Penalties**: Key claims unsupported = -0.5 penalty, supporting details = 0.0
- **Failure Categorization**: Automatic diagnosis (hallucination, weak retrieval, partial context, vague questions)
- **Retrieval Metrics**: Precision/recall tracking with missing/irrelevant document identification
- **Continuous Monitoring**: Automated evaluation suite with 55 test questions

## Key Features
- Semantic search over internal documents
- Source-cited answers
- Guardrails against hallucination
- Evaluation-driven quality checks

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

### Running Evaluations

The system includes a comprehensive evaluation framework to assess RAG performance.

1. **Start the API server:**
   ```bash
   OPENAI_API_KEY="your-key-here" python main.py
   ```

2. **Run evaluations:**
   ```bash
   python evals/run_evals.py
   ```

   This will:
   - Load questions from `evals/questions.jsonl`
   - Query the `/ask` endpoint
   - Compute metrics
   - Save detailed results to `evals/results.json`

### Evaluation Dataset

Located in `evals/questions.jsonl`, contains 55+ test questions with:

```json
{"question": "What is the purpose of the access control policy?", "expected_answer": "This policy defines requirements for managing logical access to organizational systems and data.", "source_doc_id": "access_control_policy.md"}
```

**Fields:**
- `question`: The question text
- `expected_answer`: Expected answer (optional, for groundedness evaluation)
- `source_doc_id`: Expected source document(s) (string or list)

### Evaluation Metrics

The system computes comprehensive metrics across retrieval quality, answer grounding, and overall reliability:

#### Retrieval Metrics
| Metric | Definition | Target | Why It Matters |
|--------|-----------|--------|----------------|
| **Retrieval Hit Rate** | % of questions where expected source appears in retrieved sources | > 80% | Ensures critical documents are found |
| **Retrieval Precision** | Average fraction of retrieved sources that are relevant | > 0.8 | Minimizes irrelevant document retrieval |
| **Retrieval Recall** | Average fraction of expected sources that were retrieved | > 0.8 | Ensures comprehensive context coverage |

#### Answer Quality Metrics
| Metric | Definition | Target | Why It Matters |
|--------|-----------|--------|----------------|
| **Groundedness Rate** | % of answers where weighted support score ≥70% | > 70% | Prevents hallucination and fabrication |
| **Citation Rate** | % of answers that include source citations | > 90% | Enables answer verification |
| **Answer Relevance Rate** | % of answers semantically relevant to the question | > 80% | Ensures on-topic responses |

#### Groundedness Scoring Algorithm
- **Semantic Similarity**: Uses SentenceTransformer embeddings to compare answer claims against retrieved chunks
- **Support Levels**: 
  - **Strong** (≥0.75 similarity): Full credit (weight: +1.0)
  - **Weak** (0.6-0.75 similarity): Partial credit (weight: +0.5) 
  - **Unsupported** (<0.6 similarity): No credit (weight: 0.0 or -0.5 penalty)
- **Importance Weighting**: LLM classifies sentences as 'key' (factual claims requiring strong support) vs 'supporting' (details)
- **Penalty System**: Unsupported key claims = -0.5 penalty, supporting details = 0.0 (no penalty)
- **Final Score**: Normalized to 0-100 scale accounting for penalties

#### Failure Analysis (Groundedness < 70%)
Automatic categorization of failures with actionable insights:

| Failure Type | Description | Debugging Action |
|-------------|-------------|------------------|
| **hallucination** | Unsupported key factual claims | Check chunk relevance, adjust similarity thresholds, improve prompt engineering |
| **weak_retrieval** | No relevant chunks retrieved | Enhance embedding quality, tune retrieval parameters, expand document coverage |
| **partial_context** | Missing critical expected sources (recall < 0.5) | Increase top-k retrieval, improve ranking algorithms |
| **vague_question** | Question too broad or ambiguous | Add question preprocessing, implement clarification prompts |

### Example Failure Analysis

**Question**: "What are the requirements for incident response team composition?"

**System Answer**: "The incident response team must include a team lead, technical experts, and legal counsel. Response time should be within 1 hour."

**Groundedness Score**: 45% (FAIL)

**Failure Type**: hallucination

**Unsupported Claims**:
- "Response time should be within 1 hour" (importance: key)
  - Closest chunk: "The incident response plan outlines team roles and responsibilities."
  - Similarity: 0.234 (unsupported)
- "The incident response team must include legal counsel" (importance: key)  
  - Closest chunk: "Team composition includes technical staff and management representatives."
  - Similarity: 0.412 (unsupported)

**Debugging Insights**:
- **Root Cause**: Retrieved chunks missing specific composition requirements
- **Action Items**: 
  - Verify incident response document indexing
  - Consider increasing retrieval top-k from 4 to 6
  - Review embedding model for domain-specific terms

### Results

Results are saved to `evals/results.json` with detailed per-question metrics and response data.

**How It Works:** Converts Chroma L2 distance to similarity (`similarity = 1/(1 + distance)`), compares top score against 0.35 threshold. Threshold chosen as balance: high enough for meaningful matches, low enough to reject obviously irrelevant retrievals.

**Impact:** Forces refusal for weak semantic matches while preserving existing confidence logic for stronger retrievals.

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
- **Retrieval quality**: Current evaluation shows 0% retrieval hit rate - needs tuning

### LLM Integration
- **API dependency**: Requires OpenAI API access and credits
- **Rate limits**: Subject to OpenAI API constraints
- **Cost**: Per-token pricing for answer generation
- **No offline mode**: Cannot function without internet/API access

### Enterprise Features
- **No authentication**: No user management or access controls
- **Single user**: Not designed for concurrent usage
- **Traceability**: Query and answer traces are persisted to `/traces/traces.db`
- **Trace inspection endpoint**: `GET /traces/recent` returns the latest 20 request traces
- **Debug mode**: Enable with `DEBUG_MODE=true` to print a trace summary and retrieved chunk scores per request
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
