from typing import List, Optional

import faiss
from sentence_transformers import SentenceTransformer

from src.embeddings import embed_texts
from src.models import Chunk, SearchResult
from src.config import TOP_K


def chunk_matches_filters(
    chunk: Chunk,
    document_type: Optional[str] = None,
    document_ids: Optional[List[str]] = None,
) -> bool:
    """
    Return True if a chunk satisfies the requested metadata filters.
    """

    if document_type is not None:
        if chunk.document_type != document_type:
            return False

    if document_ids is not None:
        if chunk.document_id not in document_ids:
            return False

    return True


def retrieve_chunks(
    query: str,
    model: SentenceTransformer,
    index: faiss.Index,
    chunks: List[Chunk],
    top_k: int = TOP_K,
    document_type: Optional[str] = None,
    document_ids: Optional[List[str]] = None,
    candidate_multiplier: int = 5,
) -> List[SearchResult]:
    """
    Retrieve semantically relevant chunks using FAISS.

    Optional metadata filters allow retrieval from:
    - only the thesis
    - only papers
    - specific documents
    """

    if not query.strip():
        raise ValueError("query cannot be empty")

    if top_k <= 0:
        raise ValueError("top_k must be greater than 0")

    if candidate_multiplier <= 0:
        raise ValueError(
            "candidate_multiplier must be greater than 0"
        )

    if index.ntotal != len(chunks):
        raise ValueError(
            "FAISS index and chunk metadata are not aligned"
        )

    query_embedding = embed_texts(
        texts=[query],
        model=model,
    )

    candidate_k = min(
        top_k * candidate_multiplier,
        index.ntotal,
    )

    scores, indices = index.search(
        query_embedding,
        candidate_k,
    )

    results: List[SearchResult] = []

    for score, index_position in zip(
        scores[0],
        indices[0],
    ):

        if index_position < 0:
            continue

        chunk = chunks[int(index_position)]

        if not chunk_matches_filters(
            chunk=chunk,
            document_type=document_type,
            document_ids=document_ids,
        ):
            continue

        results.append(
            SearchResult(
                chunk=chunk,
                score=float(score),
            )
        )

        if len(results) == top_k:
            break

    return results


if __name__ == "__main__":

    from src.embeddings import load_embedding_model
    from src.vector_store import load_vector_store

    model = load_embedding_model()

    index, chunks = load_vector_store()

    query = (
        "Does sentiment improve short-term volatility forecasting?"
    )

    results = retrieve_chunks(
        query=query,
        model=model,
        index=index,
        chunks=chunks,
        top_k=5,
        document_type="thesis",
    )

    print(f"Query: {query}")
    print()

    for rank, result in enumerate(
        results,
        start=1,
    ):

        chunk = result.chunk

        print(f"Result {rank}")
        print(f"Score: {result.score:.4f}")
        print(f"Document: {chunk.document_name}")
        print(f"Document type: {chunk.document_type}")
        print(f"Page: {chunk.page}")
        print(f"Section: {chunk.section}")
        print(f"Text: {chunk.text[:500]}")
        print("-" * 80)