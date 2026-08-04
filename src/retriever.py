from typing import List

import faiss
from sentence_transformers import SentenceTransformer

from embeddings import embed_texts


def retrieve_chunks(
    query: str,
    model: SentenceTransformer,
    index: faiss.Index,
    chunks: List[dict],
    top_k: int = 5,
) -> List[dict]:
    """
    Retrieve the most relevant chunks for a query.
    """

    if not query.strip():
        raise ValueError("query cannot be empty")

    if top_k <= 0:
        raise ValueError("top_k must be greater than 0")

    top_k = min(top_k, index.ntotal)

    # Trasforma la domanda in un vettore della stessa dimensione dei chunk:
    query_embedding = embed_texts(
        texts=[query],
        model=model,
    )

    # scores  = similarità (non è una probabilità ma cosine similarity:
    # * vicino a 1 → molto simile;
    # * vicino a 0 → poca relazione;
    # * negativo → significato potenzialmente opposto o molto distante.
    # indices = posizioni dei chunk
    scores, indices = index.search(
        query_embedding,
        top_k,
    )

    results = []

    for score, index_position in zip(scores[0], indices[0]):
        # recuperiamo testo e metadati e aggiungiamo il punteggio di similarità.
        chunk = chunks[index_position].copy()
        chunk["score"] = float(score)
        results.append(chunk)

    return results


if __name__ == "__main__":
    from embeddings import load_embedding_model
    from vector_store import load_vector_store

    model = load_embedding_model()
    index, chunks = load_vector_store()
    query = "Why was FinBERT used for sentiment analysis?"

    results = retrieve_chunks(
        query=query,
        model=model,
        index=index,
        chunks=chunks,
        top_k=5,
    )

    print(f"Query: {query}")
    print()

    for rank, result in enumerate(results, start=1):
        print(f"Result {rank}")
        print(f"Score: {result['score']:.4f}")
        print(f"Document: {result['document']}")
        print(f"Page: {result['page']}")
        print(f"Text: {result['text'][:500]}")
        print("-" * 80)