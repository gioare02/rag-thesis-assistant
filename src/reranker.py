from typing import List

from sentence_transformers import CrossEncoder

from src.config import RERANKER_MODEL_NAME
from src.models import SearchResult


def load_reranker(
    model_name: str = RERANKER_MODEL_NAME,
) -> CrossEncoder:
    """
    Load the CrossEncoder used to rerank
    retrieved candidate chunks.
    """

    return CrossEncoder(model_name)


def rerank_results(
    query: str,
    results: List[SearchResult],
    reranker: CrossEncoder,
    top_k: int = 5,
) -> List[SearchResult]:
    """
    Rerank retrieved candidates using a CrossEncoder.

    Each candidate is evaluated jointly with the query:

        (query, chunk text) -> relevance score

    The final results are sorted by CrossEncoder score.
    """

    if not query.strip():
        raise ValueError(
            "query cannot be empty"
        )

    if top_k <= 0:
        raise ValueError(
            "top_k must be greater than 0"
        )

    if not results:
        return []

    pairs = [
        (
            query,
            result.chunk.text,
        )
        for result in results
    ]

    scores = reranker.predict(
        pairs,
        show_progress_bar=False,
    )

    reranked_results = []

    for result, score in zip(
        results,
        scores,
    ):

        reranked_results.append(
            SearchResult(
                chunk=result.chunk,
                score=float(score),
            )
        )

    reranked_results.sort(
        key=lambda result: result.score,
        reverse=True,
    )

    return reranked_results[:top_k]


if __name__ == "__main__":

    from src.embeddings import (
        load_embedding_model,
    )

    from src.vector_store import (
        load_vector_store,
    )

    from src.hybrid_retriever import (
        hybrid_retrieve,
    )

    index, chunks = load_vector_store()

    embedding_model = (
        load_embedding_model()
    )

    reranker = load_reranker()

    query = (
        "Does sentiment improve "
        "short-term volatility forecasting?"
    )

    hybrid_results = hybrid_retrieve(
        query=query,
        model=embedding_model,
        index=index,
        chunks=chunks,
        top_k=15,
        document_type="thesis",
    )

    reranked_results = rerank_results(
        query=query,
        results=hybrid_results,
        reranker=reranker,
        top_k=5,
    )

    print(f"Query: {query}")
    print()

    print("HYBRID CANDIDATES")
    print("=" * 80)

    for rank, result in enumerate(
        hybrid_results,
        start=1,
    ):

        chunk = result.chunk

        print(
            f"{rank}. "
            f"RRF={result.score:.6f} | "
            f"Page={chunk.page} | "
            f"Section={chunk.section}"
        )

    print()
    print("RERANKED RESULTS")
    print("=" * 80)

    for rank, result in enumerate(
        reranked_results,
        start=1,
    ):

        chunk = result.chunk

        print(f"Result {rank}")
        print(
            f"CrossEncoder score: "
            f"{result.score:.4f}"
        )
        print(
            f"Document: "
            f"{chunk.document_name}"
        )
        print(
            f"Type: "
            f"{chunk.document_type}"
        )
        print(
            f"Page: "
            f"{chunk.page}"
        )
        print(
            f"Section: "
            f"{chunk.section}"
        )
        print(
            f"Text: "
            f"{chunk.text[:500]}"
        )
        print("-" * 80)