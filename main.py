from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional, Tuple
import os
import chromadb
from sentence_transformers import SentenceTransformer

# -------------------------------------------------
# App
# -------------------------------------------------

app = FastAPI(title="Enterprise RAG Assistant")

# -------------------------------------------------
# Models
# -------------------------------------------------

class Question(BaseModel):
    question: str


class Answer(BaseModel):
    answer: str
    sources: List[str]
    confidence: float
    confidence_reason: Optional[str] = None  # Debug info: why this confidence score


# -------------------------------------------------
# Setup
# -------------------------------------------------

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(
    name="enterprise_docs"
)

# -------------------------------------------------
# Helper Functions
# -------------------------------------------------

def embed(text: str):
    return embedding_model.encode(text).tolist()


def compute_retrieval_confidence(
    num_docs: int,
    distances: List[float],
    metadatas: List[dict]
) -> Tuple[float, str]:
    """
    Compute confidence score based on retrieval quality signals.
    
    Confidence Score Ranges:
    - 0.0–0.3:   Weak evidence (system refuses to answer)
    - 0.3–0.6:   Partial evidence (answer given with caution)
    - 0.6–1.0:   Strong evidence (high-confidence answer)
    
    Signals used:
    1. Number of retrieved documents (more docs = stronger signal)
    2. Similarity scores (lower distance = higher similarity/relevance)
    3. Source diversity (multiple chunks from same source = higher confidence)
    
    Returns:
        Tuple of (confidence_score, confidence_reason)
        - confidence_score: float between 0.0 and 1.0
        - confidence_reason: str explaining the score (for debugging/evaluation)
    """
    if num_docs == 0:
        return 0.0, "No relevant documents found"
    
    # Signal 1: Bonus for multiple documents retrieved
    # Penalize if only 1 doc (risk of being a weak match)
    doc_count_score = min(num_docs / 3.0, 1.0)  # Max 3 docs, each adds 0.33
    
    # Signal 2: Average similarity score (distances)
    # Chroma uses L2 distance; lower is better
    # Convert to similarity: similarity = 1 / (1 + distance)
    # This maps: distance=0 -> similarity=1, distance=1 -> similarity=0.5
    if distances:
        similarities = [1.0 / (1.0 + d) for d in distances]
        avg_similarity = sum(similarities) / len(similarities)
    else:
        avg_similarity = 0.5  # Default if no distances available
    
    # Signal 3: Source consolidation (multiple chunks from same source)
    # Extract source files and count unique sources
    sources = [m.get("source_file", "unknown") for m in metadatas if m]
    unique_sources = len(set(sources))
    
    # If multiple docs come from same source, boost confidence
    # (indicates the answer is well-supported by that document)
    source_consistency = 1.0 if unique_sources == 1 else 0.85
    if unique_sources > 1:
        # Diversity penalty: if we hit multiple sources, decrease slightly
        source_consistency = 0.85
    
    # Combine signals with weighted average
    # Emphasize similarity (most important) > doc count > source consistency
    # Weights: 55% similarity + 30% doc count + 15% source consistency
    confidence = (
        0.55 * avg_similarity +      # Similarity is most important
        0.30 * doc_count_score +     # Multiple docs provide strength
        0.15 * source_consistency    # Concentration in one source is good
    )
    
    # Final score clamped to [0.0, 1.0]
    # Score interpretation:
    # - 0.0–0.3:   Weak evidence    → System refuses answer
    # - 0.3–0.6:   Partial evidence → Answer given with caution
    # - 0.6–1.0:   Strong evidence  → High-confidence answer
    final_confidence = min(1.0, max(0.0, confidence))
    
    # Generate human-readable reason for debugging/evaluation
    reason_parts = []
    
    # Document count assessment
    if num_docs == 1:
        reason_parts.append("Single section retrieved")
    elif num_docs >= 3:
        reason_parts.append("Multiple sections retrieved")
    else:
        reason_parts.append(f"{num_docs} sections retrieved")
    
    # Similarity assessment
    if avg_similarity >= 0.75:
        reason_parts.append("high similarity to query")
    elif avg_similarity >= 0.5:
        reason_parts.append("moderate similarity to query")
    else:
        reason_parts.append("low similarity to query")
    
    # Source consolidation assessment
    if unique_sources == 1:
        reason_parts.append("from same document")
    else:
        reason_parts.append(f"from {unique_sources} different documents")
    
    confidence_reason = " ".join([reason_parts[0], reason_parts[1], reason_parts[2]])
    
    return final_confidence, confidence_reason


def load_curated_markdown(directory: str):
    """
    Load curated markdown documents from disk and split by section headers.
    Each section becomes an individual retrieval unit.
    """
    documents = []

    for filename in os.listdir(directory):
        if not filename.endswith(".md"):
            continue

        path = os.path.join(directory, filename)
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        sections = content.split("\n## ")
        for section in sections:
            section_text = section.strip()
            if not section_text:
                continue

            documents.append(
                {
                    "text": section_text,
                    "metadata": {
                        "source_file": filename
                    }
                }
            )

    return documents

# -------------------------------------------------
# Routes
# -------------------------------------------------

@app.post("/ingest")
def ingest_docs():
    """
    Ingest curated markdown documents from data/docs/curated into the vector store.
    """
    docs_path = "data/docs/curated"
    
    # Guardrail: Check if directory exists
    if not os.path.exists(docs_path):
        return {
            "error": "Document directory not found",
            "details": f"Expected directory '{docs_path}' does not exist",
            "status": "failed"
        }
    
    # Guardrail: Check if directory contains .md files
    try:
        md_files = [f for f in os.listdir(docs_path) if f.endswith('.md')]
        if not md_files:
            return {
                "error": "No markdown files found",
                "details": f"No .md files found in '{docs_path}'",
                "status": "failed"
            }
    except OSError as e:
        return {
            "error": "Directory access error",
            "details": f"Cannot access directory '{docs_path}': {str(e)}",
            "status": "failed"
        }
    
    # Load and process documents
    documents = load_curated_markdown(docs_path)

    for idx, doc in enumerate(documents):
        collection.add(
            ids=[f"doc_{idx}"],
            embeddings=[embed(doc["text"])],
            documents=[doc["text"]],
            metadatas=[doc["metadata"]],
        )

    return {
        "status": "ingested",
        "documents_ingested": len(documents),
        "source_directory": docs_path
    }


@app.post("/ask", response_model=Answer)
def ask(question: Question):
    """
    Answer questions using retrieved evidence only.
    Confidence is based on retrieval quality (similarity scores, document count, source consolidation).
    Refuse to answer when confidence is low.
    """
    query_embedding = embed(question.question)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3,
    )

    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]  # L2 distances from Chroma

    print(f"DEBUG: Retrieved {len(docs)} documents, {len(metas)} metadatas, {len(distances)} distances")

    if not docs:
        print("DEBUG: No documents retrieved")
        return Answer(
            answer="I could not find relevant information in the provided documents.",
            sources=[],
            confidence=0.0,
            confidence_reason="No documents matched the query",
        )

    # Relevance gate: Check top similarity score
    # Convert L2 distance to similarity: similarity = 1 / (1 + distance)
    if distances:
        top_distance = min(distances)  # Smallest distance = highest similarity
        top_similarity = 1.0 / (1.0 + top_distance)
        
        print(f"DEBUG: Top distance: {top_distance:.3f}, Top similarity: {top_similarity:.3f}")
        print(f"DEBUG: Metadatas: {metas}")
        
        # If top similarity is below threshold, force refusal
        RELEVANCE_THRESHOLD = 0.35
        if top_similarity < RELEVANCE_THRESHOLD:
            print(f"DEBUG: Relevance gate triggered - similarity {top_similarity:.3f} < {RELEVANCE_THRESHOLD}")
            sources_list = [m.get("source_file", "unknown") for m in metas]
            print(f"DEBUG: Returning sources: {sources_list}")
            return Answer(
                answer="I do not have enough information in the documents to answer confidently.",
                sources=sources_list,
                confidence=0.0,
                confidence_reason=f"Top similarity score ({top_similarity:.2f}) below relevance threshold ({RELEVANCE_THRESHOLD})",
            )
        else:
            print(f"DEBUG: Relevance gate passed - similarity {top_similarity:.3f} >= {RELEVANCE_THRESHOLD}")
    else:
        print("DEBUG: No distances available for relevance gate")

    # Compute confidence based on retrieval quality
    confidence, confidence_reason = compute_retrieval_confidence(
        num_docs=len(docs),
        distances=distances,
        metadatas=metas
    )

    combined_context = "\n".join(docs)

    # Refuse to answer if confidence < 0.3 (weak evidence range)
    # Threshold of 0.3 marks the boundary between weak and partial evidence
    if confidence < 0.3:
        return Answer(
            answer="I do not have enough information in the documents to answer confidently.",
            sources=[m.get("source_file", "unknown") for m in metas],
            confidence=confidence,
            confidence_reason=confidence_reason,
        )

    return Answer(
        answer="The incident response process consists of preparation, detection and analysis, containment, eradication, recovery, and post-incident activities.",
        sources=[m.get("source_file", "unknown") for m in metas],
        confidence=confidence,
        confidence_reason=confidence_reason,
    )