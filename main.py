from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
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


# -------------------------------------------------
# Setup
# -------------------------------------------------

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(
    name="enterprise_docs"
)

# -------------------------------------------------
# Helper Functions
# -------------------------------------------------

def embed(text: str):
    return embedding_model.encode(text).tolist()


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
    Ingest curated markdown documents from docs/curated into the vector store.
    """
    docs_path = "docs/curated"
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
    }


@app.post("/ask", response_model=Answer)
def ask(question: Question):
    """
    Answer questions using retrieved evidence only.
    Refuse to answer when confidence is low.
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

    # Simple heuristic confidence score
    confidence = min(1.0, len(combined_context) / 500)

    if confidence < 0.3:
        return Answer(
            answer="I do not have enough information in the documents to answer confidently.",
            sources=[m.get("source_file", "unknown") for m in metas],
            confidence=confidence,
        )

    return Answer(
        answer=f"Based on the documents:\n\n{combined_context}",
        sources=[m.get("source_file", "unknown") for m in metas],
        confidence=confidence,
    )