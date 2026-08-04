from collections import defaultdict
from typing import Dict, List

from langchain_core.documents import Document

from src.models import Chunk, SearchResult


def fuse_results_rrf(
    semantic_results: List[SearchResult],
    lexical_results: List[Document],
    top_k: int = 5,
    rrf_constant: int = 60,
) -> List[SearchResult]:
    """
    Combine FAISS and LangChain BM25 rankings using
    Reciprocal Rank Fusion.
    """

    if top_k <= 0:
        raise ValueError("top_k must be greater than 0")

    if rrf_constant <= 0:
        raise ValueError("rrf_constant must be greater than 0")

    scores: Dict[int, float] = defaultdict(float)
    chunks_by_id: Dict[int, Chunk] = {}

    for rank, result in enumerate(
        semantic_results,
        start=1,
    ):
        chunk = result.chunk
        chunk_id = chunk.chunk_id

        scores[chunk_id] += 1 / (rrf_constant + rank)
        chunks_by_id[chunk_id] = chunk

    for rank, document in enumerate(
        lexical_results,
        start=1,
    ):
        chunk_id = int(document.metadata["chunk_id"])

        scores[chunk_id] += 1 / (rrf_constant + rank)

        if chunk_id not in chunks_by_id:
            chunks_by_id[chunk_id] = Chunk(
                chunk_id=chunk_id,
                document=str(document.metadata["document"]),
                page=int(document.metadata["page"]),
                text=document.page_content,
            )

    ranked_chunk_ids = sorted(
        scores,
        key=lambda chunk_id: scores[chunk_id],
        reverse=True,
    )

    return [
        SearchResult(
            chunk=chunks_by_id[chunk_id],
            score=scores[chunk_id],
        )
        for chunk_id in ranked_chunk_ids[:top_k]
    ]


######################### TEST ##########################

if __name__ == "__main__":
    from src.bm25_retriever import build_bm25_retriever
    from src.embeddings import load_embedding_model
    from src.langchain_documents import chunks_to_documents
    from src.retriever import retrieve_chunks
    from src.vector_store import load_vector_store

    index, chunks = load_vector_store()
    embedding_model = load_embedding_model()
    documents = chunks_to_documents(chunks)

    bm25_retriever = build_bm25_retriever(
        documents=documents,
        top_k=10,
    )

    query = "CONF 0.97 KNN 0.90"

    semantic_results = retrieve_chunks(
        query=query,
        model=embedding_model,
        index=index,
        chunks=chunks,
        top_k=10,
    )

    lexical_results = bm25_retriever.invoke(query)

    hybrid_results = fuse_results_rrf(
        semantic_results=semantic_results,
        lexical_results=lexical_results,
        top_k=5,
    )

    for rank, result in enumerate(hybrid_results, start=1):
        print(f"Result {rank}")
        print(f"RRF score: {result.score:.6f}")
        print(f"Document: {result.chunk.document}")
        print(f"Page: {result.chunk.page}")
        print(result.chunk.text[:500])
        print("-" * 80)