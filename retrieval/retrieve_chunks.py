"""
Semantic retrieval from Chroma collection.

Queries the enterprise_docs collection with semantic search,
returning the most relevant chunks based on cosine similarity.
"""

import chromadb
from sentence_transformers import SentenceTransformer
from config import TOP_K, MIN_SIMILARITY_THRESHOLD


def retrieve_similar_chunks(query: str, top_k: int = TOP_K, collection_name: str = "enterprise_docs"):
    """
    Perform semantic search against the Chroma collection.

    Args:
        query: Plain text search query
        top_k: Number of top results to return (default: 5)
        collection_name: Name of the Chroma collection

    Returns:
        List of dicts with retrieval results
    """

    # Step 1: Initialize embedding model (same as used for storage)
    print("\n" + "=" * 80)
    print("STEP 1: INITIALIZE EMBEDDING MODEL")
    print("=" * 80)

    model_name = "all-MiniLM-L6-v2"
    embedding_model = SentenceTransformer(model_name)
    print(f"✓ Loaded embedding model: {model_name}")

    # Step 2: Connect to existing Chroma collection
    print("\n" + "=" * 80)
    print("STEP 2: CONNECT TO CHROMA COLLECTION")
    print("=" * 80)

    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    print(f"✓ Initialized persistent Chroma client")

    try:
        collection = chroma_client.get_collection(name=collection_name)
        print(f"✓ Connected to collection: '{collection_name}'")
    except Exception as e:
        print(f"✗ Failed to connect to collection '{collection_name}': {e}")
        print("  Make sure to run embed_and_store.py first to create the collection.")
        return []

    # Step 3: Encode query
    print("\n" + "=" * 80)
    print("STEP 3: ENCODE QUERY")
    print("=" * 80)

    query_embedding = embedding_model.encode(query)
    print(f"✓ Encoded query: '{query}'")
    print(f"  Embedding dimension: {len(query_embedding)}")

    # Step 4: Perform semantic search
    print("\n" + "=" * 80)
    print("STEP 4: PERFORM SEMANTIC SEARCH")
    print("=" * 80)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["embeddings", "documents", "metadatas", "distances"]
    )

    print(f"✓ Retrieved {len(results['ids'][0])} results")

    # Step 5: Format and display results
    print("\n" + "=" * 80)
    print("STEP 5: RESULTS")
    print("=" * 80)

    formatted_results = []

    for i, (chunk_id, distance, document, metadata) in enumerate(zip(
        results['ids'][0],
        results['distances'][0],
        results['documents'][0],
        results['metadatas'][0]
    )):
        # Convert cosine distance to similarity score (1 - distance)
        similarity_score = 1.0 - distance

        result = {
            "rank": i + 1,
            "chunk_id": chunk_id,
            "similarity_score": similarity_score,
            "section_title": metadata.get("section_title", "Unknown"),
            "source_file": metadata.get("source_file", "Unknown"),
            "text_preview": document[:200] + "..." if len(document) > 200 else document
        }

        formatted_results.append(result)

        # Print formatted result
        print(f"\n--- Result {result['rank']} ---")
        print(f"Similarity Score: {result['similarity_score']:.4f}")
        print(f"Section Title: {result['section_title']}")
        print(f"Source File: {result['source_file']}")
        print(f"Text Preview: {result['text_preview']}")

    print("\n" + "=" * 80)
    print("✓ RETRIEVAL COMPLETE")
    print("=" * 80)

    return formatted_results


if __name__ == "__main__":
    # Example usage - you can modify these queries
    test_queries = [
        "What is the incident response process?",
        "How do I manage user access controls?"
    ]

    for query in test_queries:
        print(f"\n{'='*100}")
        print(f"QUERY: {query}")
        print(f"{'='*100}")
        retrieve_similar_chunks(query, top_k=4)