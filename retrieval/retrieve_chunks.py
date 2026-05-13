"""
Semantic retrieval from Chroma collection (pure vector search).

Queries the enterprise_docs collection with semantic search,
returning the most relevant chunks based on cosine similarity.
No API calls, no LLM dependencies - pure local vector search.
"""

import chromadb
from sentence_transformers import SentenceTransformer


def retrieve_similar_chunks(
    query: str,
    top_k: int = 5,
    collection_name: str = "enterprise_docs",
    min_similarity: float = 0.4,
    verbose: bool = False
):
    """
    Perform semantic search against the Chroma collection.

    Args:
        query: Plain text search query
        top_k: Number of results to return
        collection_name: Chroma collection name
        min_similarity: Minimum similarity threshold
        verbose: Enable debug prints

    Returns:
        List of chunk dicts
    """

    # ✅ Load embedding model
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # ✅ Connect to DB
    try:
        client = chromadb.PersistentClient(path="./chroma_db")
        collection = client.get_collection(name=collection_name)
    except Exception as e:
        print(f"Error connecting to Chroma: {e}")
        return []

    # ✅ Embed query
    query_embedding = model.encode(query)

    # ✅ Get more candidates to filter later
    n_candidates = max(top_k * 3, 20)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_candidates,
        include=["documents", "metadatas", "distances"]
    )

    retrieved_chunks = []

    for chunk_id, distance, doc, metadata in zip(
        results["ids"][0],
        results["distances"][0],
        results["documents"][0],
        results["metadatas"][0]
    ):
        similarity = 1.0 - distance

        if similarity < min_similarity:
            continue

        retrieved_chunks.append({
            "text": doc,
            "source_file": metadata.get("source_file", "Unknown"),
            "section_title": metadata.get("section_title", "Unknown"),
            "similarity_score": similarity,
            "chunk_id": chunk_id
        })

    # ✅ Fallback if nothing passes threshold
    if not retrieved_chunks:
        if verbose:
            print("⚠️ No chunks passed threshold — using fallback")

        fallback_chunks = []
        for chunk_id, distance, doc, metadata in zip(
            results["ids"][0][:top_k],
            results["distances"][0][:top_k],
            results["documents"][0][:top_k],
            results["metadatas"][0][:top_k]
        ):
            fallback_chunks.append({
                "text": doc,
                "source_file": metadata.get("source_file", "Unknown"),
                "section_title": metadata.get("section_title", "Unknown"),
                "similarity_score": 1.0 - distance,
                "chunk_id": chunk_id
            })

        return fallback_chunks

    # ✅ Limit results
    top_chunks = retrieved_chunks[:top_k]

    if verbose:
        print(f"\nRetrieved {len(top_chunks)} chunks (filtered from {len(retrieved_chunks)})")
        for i, c in enumerate(top_chunks, 1):
            print(f"{i}. {c['source_file']} ({c['similarity_score']:.3f})")

    return top_chunks


if __name__ == "__main__":
    test_query = "What is incident response?"
    chunks = retrieve_similar_chunks(test_query, verbose=True)