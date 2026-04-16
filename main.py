from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import chromadb
from sentence_transformers import SentenceTransformer

app = FastAPI(title="Enterprise RAG Assistant")

# --- Models ---

class Question(BaseModel):
    question: str

class Answer(BaseModel):
    answer: str
    sources: List[str]
    confidence: float


# --- Setup ---

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(
    name="enterprise_docs"
)


# --- Helper Functions ---

def embed(text: str):
    return embedding_model.encode(text).tolist()


# --- Routes ---

@app.post("/ingest")
def ingest_docs(documents: List[str]):
    """
    Ingest documents into the knowledge base.
    """
    for i, doc in enumerate(documents):
        collection.add(
            ids=[str(i)],
            embeddings=[embed(doc)],
            documents=[doc],
            metadatas=[{"source": f"doc_{i}"}],
        )
    return {"status": "ingested", "count": len(documents)}


@app.post("/ask", response_model=Answer)
def ask(question: Question):
    """
    Answer using retrieved evidence only.
    Refuse if confidence is low.
    """
    query_embedding = embed(question.question)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3,
    )

    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]

    if not docs:
        return Answer(
            answer="I could not find relevant information in the provided documents.",
            sources=[],
            confidence=0.0,
        )

    combined_context = "\n".join(docs)

    # Simple confidence heuristic (good enough for MVP)
    confidence = min(1.0, len(combined_context) / 500)

    if confidence < 0.3:
        return Answer(
            answer="I do not have enough information to answer confidently.",
            sources=[m["source"] for m in metas],
            confidence=confidence,
        )

    return Answer(
        answer=f"Based on the documents: {combined_context}",
        sources=[m["source"] for m in metas],
        confidence=confidence,
    )
