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
    
    Pure vector search - no API calls, no external dependencies.

    Args:
        query: Plain text search query
        top_k: Number of top results to return (default: 8)
        collection_name: Name of the Chroma collection (default: "enterprise_docs")
        min_similarity: Minimum similarity threshold (0.0-1.0, default: 0.3)
                       Chunks below this threshold are filtered out
        verbose: Print detailed retrieval steps (default: False)

    Returns:
        List of dicts with keys:
        - text: Full document text
        - source_file: Source document filename
        - section_title: Section title from metadata
        - similarity_score: Cosine similarity (0.0-1.0)
        - chunk_id: Unique chunk identifier
        
        Returns empty list if retrieval fails or no chunks meet similarity threshold
    """
    
    if verbose:
        print("\n" + "=" * 80)
        print("RETRIEVAL: INITIALIZE")
        print("=" * 80)

    # Initialize embedding model (same as used for storage)
    model_name = "all-MiniLM-L6-v2"
    embedding_model = SentenceTransformer(model_name)
    
    if verbose:
        print(f"✓ Loaded embedding model: {model_name}")

    # Connect to persistent Chroma collection
    try:
        chroma_client = chromadb.PersistentClient(path="./chroma_db")
        collection = chroma_client.get_collection(name=collection_name)
        if verbose:
            print(f"✓ Connected to collection: '{collection_name}'")
    except Exception as e:
        print(f"Error: Failed to connect to collection '{collection_name}': {e}")
        print("       Make sure to run embed_and_store.py first to create the collection.")
        return []

    # Encode query to embedding
    query_embedding = embedding_model.encode(query)
    
    if verbose:
        print(f"✓ Encoded query: '{query}'")
        print(f"  Embedding dimension: {len(query_embedding)}")

    # Retrieve candidates with margin for filtering
    # Query more results than needed to account for similarity filtering
    n_candidates = max(top_k * 2, 20)
    
    if verbose:
        print(f"\n" + "=" * 80)
        print("RETRIEVAL: SEMANTIC SEARCH")
        print("=" * 80)
        print(f"  Querying for {n_candidates} candidates...")

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_candidates,
        include=["documents", "metadatas", "distances"]
    )

    # Convert distances to similarity scores and filter
    # Distance = 1 - cosine_similarity, so similarity = 1 - distance
    retrieved_chunks = []
    
    for chunk_id, distance, document, metadata in zip(
        results['ids'][0],
        results['distances'][0],
        results['documents'][0],
        results['metadatas'][0]
    ):
        similarity_score = 1.0 - distance
        
        # Filter by similarity threshold
        if similarity_score < min_similarity:
            continue
        
        chunk = {
            "text": document,
            "source_file": metadata.get("source_file", "Unknown"),
            "section_title": metadata.get("section_title", "Unknown"),
            "similarity_score": similarity_score,
            "chunk_id": chunk_id
        }
        retrieved_chunks.append(chunk)

    # Limit to top_k results
    top_chunks = retrieved_chunks[:top_k]
    
    if verbose:
        print(f"✓ Retrieved {len(top_chunks)}/{len(retrieved_chunks)} chunks")
        print(f"  (Filtered by min_similarity >= {min_similarity})")
        print(f"\n" + "=" * 80)
        print("RETRIEVAL: RESULTS")
        print("=" * 80)
        
        for i, chunk in enumerate(top_chunks, 1):
            print(f"\n--- Result {i} ---")
            print(f"Similarity: {chunk['similarity_score']:.4f}")
            print(f"Source: {chunk['source_file']}")
            print(f"Section: {chunk['section_title']}")
            text_preview = chunk['text'][:150] + "..." if len(chunk['text']) > 150 else chunk['text']
            print(f"Text: {text_preview}")
        
        print("\n" + "=" * 80)

    return top_chunks


if __name__ == "__main__":
    # Example usage - test queries with verbose output
    test_queries = [
        "What is the incident response process?",
        "How do I manage user access controls?"
    ]

    for query in test_queries:
        print(f"\n{'='*100}")
        print(f"QUERY: {query}")
        print(f"{'='*100}")
        chunks = retrieve_similar_chunks(query, top_k=8, verbose=True)