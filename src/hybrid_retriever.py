from typing import List, Optional, Dict

import faiss
from sentence_transformers import SentenceTransformer

from src.models import Chunk, SearchResult
from src.retriever import retrieve_chunks
from src.bm25_retriever import retrieve_bm25
from src.config import TOP_K


def reciprocal_rank_fusion(
    result_lists: List[List[SearchResult]],
    top_k: int = TOP_K,
    rrf_k: int = 60,
) -> List[SearchResult]:
    """
    Combine multiple ranked retrieval result lists using
    Reciprocal Rank Fusion (RRF).

    RRF score:

        score(d) = sum 1 / (rrf_k + rank)

    where the sum is computed across all retrievers
    that returned the document.
    """

    if top_k <= 0:
        raise ValueError(
            "top_k must be greater than 0"
        )

    if rrf_k <= 0:
        raise ValueError(
            "rrf_k must be greater than 0"
        )

    fused_scores: Dict[int, float] = {}

    chunk_lookup: Dict[int, Chunk] = {}

    for results in result_lists:

        for rank, result in enumerate(
            results,
            start=1,
        ):

            chunk_id = result.chunk.chunk_id

            rrf_score = 1.0 / (
                rrf_k + rank
            )

            if chunk_id not in fused_scores:
                fused_scores[chunk_id] = 0.0

            fused_scores[chunk_id] += rrf_score

            chunk_lookup[chunk_id] = result.chunk

    ranked_chunk_ids = sorted(
        fused_scores,
        key=fused_scores.get,
        reverse=True,
    )

    fused_results = []

    for chunk_id in ranked_chunk_ids[:top_k]:

        fused_results.append(
            SearchResult(
                chunk=chunk_lookup[chunk_id],
                score=fused_scores[chunk_id],
            )
        )

    return fused_results


def hybrid_retrieve(
    query: str,
    model: SentenceTransformer,
    index: faiss.Index,
    chunks: List[Chunk],
    top_k: int = TOP_K,
    document_type: Optional[str] = None,
    document_ids: Optional[List[str]] = None,
    per_retriever_k: Optional[int] = None,
    rrf_k: int = 60,
) -> List[SearchResult]:
    """
    Perform hybrid retrieval using:

    1. Dense semantic retrieval with FAISS
    2. Lexical retrieval with BM25
    3. Reciprocal Rank Fusion

    Metadata filters are passed consistently to
    both retrievers.
    """

    if not query.strip():
        raise ValueError(
            "query cannot be empty"
        )

    if top_k <= 0:
        raise ValueError(
            "top_k must be greater than 0"
        )

    if per_retriever_k is None:

        per_retriever_k = max(
            top_k * 3,
            top_k,
        )

    dense_results = retrieve_chunks(
        query=query,
        model=model,
        index=index,
        chunks=chunks,
        top_k=per_retriever_k,
        document_type=document_type,
        document_ids=document_ids,
    )

    bm25_results = retrieve_bm25(
        query=query,
        chunks=chunks,
        top_k=per_retriever_k,
        document_type=document_type,
        document_ids=document_ids,
    )

    fused_results = reciprocal_rank_fusion(
        result_lists=[
            dense_results,
            bm25_results,
        ],
        top_k=top_k,
        rrf_k=rrf_k,
    )

    return fused_results


if __name__ == "__main__":

    from src.embeddings import load_embedding_model
    from src.vector_store import load_vector_store

    model = load_embedding_model()

    index, chunks = load_vector_store()

    query = (
        "Does sentiment improve short-term "
        "volatility forecasting?"
    )

    results = hybrid_retrieve(
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
        print(
            f"RRF score: {result.score:.6f}"
        )
        print(
            f"Document: {chunk.document_name}"
        )
        print(
            f"Type: {chunk.document_type}"
        )
        print(
            f"Page: {chunk.page}"
        )
        print(
            f"Section: {chunk.section}"
        )
        print(
            f"Text: {chunk.text[:500]}"
        )
        print("-" * 80)