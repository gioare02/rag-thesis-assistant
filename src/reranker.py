from typing import List

from sentence_transformers import CrossEncoder

from src.config import RERANKER_MODEL_NAME
from src.models import SearchResult


def load_reranker(
    model_name: str = RERANKER_MODEL_NAME,
) -> CrossEncoder:
    """
    Load the CrossEncoder used to rerank retrieved chunks.
    """

    return CrossEncoder(model_name)


def rerank_results(
    query: str,
    results: List[SearchResult],
    reranker: CrossEncoder,
    top_k: int = 5,
) -> List[SearchResult]:
    """
    Rerank retrieved results using a CrossEncoder.

    The model evaluates each (query, chunk) pair jointly and
    returns a relevance score.
    """

    if not query.strip():
        raise ValueError("query cannot be empty")
    if top_k <= 0:
        raise ValueError("top_k must be greater than 0")
    if not results:
        return []

    pairs = [
        (query, result.chunk.text)
        for result in results
    ]
    scores = reranker.predict(
        pairs,
        show_progress_bar=False,
    )
    reranked_results = [
        SearchResult(
            chunk=result.chunk,
            score=float(score),
        )
        for result, score in zip(results, scores)
    ]
    reranked_results.sort(
        key=lambda result: result.score,
        reverse=True,
    )
    return reranked_results[:top_k]



######################### TEST ##########################

if __name__ == "__main__":
    from src.bm25_retriever import build_bm25_retriever
    from src.embeddings import load_embedding_model
    from src.hybrid_retriever import fuse_results_rrf
    from src.langchain_documents import chunks_to_documents
    from src.retriever import retrieve_chunks
    from src.vector_store import load_vector_store

    index, chunks = load_vector_store()

    embedding_model = load_embedding_model()
    reranker = load_reranker()

    documents = chunks_to_documents(chunks)

    bm25_retriever = build_bm25_retriever(
        documents=documents,
        top_k=10,
    )

    query = "Why was FinBERT used in the thesis?"

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
        top_k=10,
    )

    reranked_results = rerank_results(
        query=query,
        results=hybrid_results,
        reranker=reranker,
        top_k=5,
    )

    for rank, result in enumerate(reranked_results, start=1):
        print(f"Result {rank}")
        print(f"Reranker score: {result.score:.6f}")
        print(f"Document: {result.chunk.document}")
        print(f"Page: {result.chunk.page}")
        print(result.chunk.text[:500])
        print("-" * 80)