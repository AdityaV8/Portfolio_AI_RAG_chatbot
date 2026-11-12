import math
import sys
from pymongo.errors import OperationFailure

from database.connection import get_collection
from config.settings import (
    VECTOR_INDEX_NAME,
    SIMILARITY_THRESHOLD
)
from llm.providers import create_embedding

# MongoDB collection (defaults to documents_jina)
collection = get_collection()


def create_query_embedding(query):
    """Convert the user's question into a Jina vector embedding."""
    return create_embedding(query, is_query=True)


def _cosine_similarity(v1, v2):
    """Compute cosine similarity between two vectors."""
    dot = sum(a * b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a * a for a in v1))
    norm2 = math.sqrt(sum(b * b for b in v2))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)


def _vector_search_local(query_embedding, limit=3):
    """
    Fallback vector search in Python for local MongoDB instances
    that do not support Atlas $vectorSearch (e.g. SearchNotEnabled).
    """
    cursor = collection.find(
        {"embedding": {"$exists": True}},
        {"_id": 0, "text": 1, "metadata": 1, "embedding": 1}
    )
    docs = list(cursor)
    if not docs:
        print(
            "Warning: No documents found with embeddings in the collection. "
            "Did you run 'python -m ingestion.ingest'?",
            file=sys.stderr
        )
        return []

    scored = []
    for doc in docs:
        emb = doc.get("embedding")
        if not emb:
            continue
        # Convert cosine similarity (-1 to 1) to Atlas-style score (0 to 1): (1 + cos) / 2
        cos_sim = _cosine_similarity(query_embedding, emb)
        score = (1.0 + cos_sim) / 2.0
        scored.append({
            "text": doc.get("text", ""),
            "metadata": doc.get("metadata", {}),
            "score": score
        })

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:limit]


def retrieve_documents(query, limit=3):
    """
    Search MongoDB for documents similar to the user's question.
    Attempts Atlas $vectorSearch first; falls back to local in-memory
    vector search if running against standard MongoDB Community Server.

    Returns:
        matching documents if the similarity threshold is satisfied
        otherwise None
    """
    # Step 1: Convert question into embedding
    query_embedding = create_query_embedding(query)

    # Step 2: Vector search (try Atlas $vectorSearch, fallback to local search)
    try:
        results = collection.aggregate([
            {
                "$vectorSearch": {
                    "index": VECTOR_INDEX_NAME,
                    "path": "embedding",
                    "queryVector": query_embedding,
                    "numCandidates": 10,
                    "limit": limit
                }
            },
            {
                "$project": {
                    "_id": 0,
                    "text": 1,
                    "metadata": 1,
                    "score": {
                        "$meta": "vectorSearchScore"
                    }
                }
            }
        ])
        results = list(results)
    except OperationFailure as e:
        print(
            f"Atlas $vectorSearch not available ({e.details.get('codeName', e.args[0] if e.args else e)}). "
            f"Falling back to local Python cosine similarity.",
            file=sys.stderr
        )
        results = _vector_search_local(query_embedding, limit=limit)

    # Step 3: No results
    if not results:
        return None

    # Step 4: Check the best matching document
    best_score = results[0]["score"]

    print(f"Best similarity score: {best_score:.4f}")

    if best_score < SIMILARITY_THRESHOLD:
        return None

    # Step 5: Return relevant documents
    return results


if __name__ == "__main__":
    question = input("Enter your question: ")
    results = retrieve_documents(question)

    if results is None:
        print("\nInformation not available in the knowledge base.")
        print("This information will be discovered and updated further.")
    else:
        print("\nRetrieved Documents:")
        for result in results:
            print("\n-------------------------")
            print("Score:", result["score"])
            print("Source:", result.get("metadata", {}).get("source", "unknown"))
            print("Section:", result.get("metadata", {}).get("section", "unknown"))
            print("Text:")
            print(result["text"])