from typing import List, Dict

from src.models import Chunk, SearchResult


def build_chunk_lookup(
    chunks: List[Chunk],
) -> Dict[int, Chunk]:
    """
    Build a dictionary for fast chunk lookup by chunk_id.
    """

    return {
        chunk.chunk_id: chunk
        for chunk in chunks
    }


def is_valid_neighbour(
    center_chunk: Chunk,
    neighbour_chunk: Chunk,
) -> bool:
    """
    A neighbour is valid only if it belongs to:
    - the same document
    - the same section
    """

    same_document = (
        center_chunk.document_id
        == neighbour_chunk.document_id
    )

    same_section = (
        center_chunk.section
        == neighbour_chunk.section
    )

    return same_document and same_section


def expand_single_result(
    result: SearchResult,
    chunk_lookup: Dict[int, Chunk],
    include_previous: bool = True,
    include_next: bool = True,
) -> List[Chunk]:
    """
    Expand one retrieved result with neighbouring chunks.

    Order:
        previous -> center -> next
    """

    center_chunk = result.chunk

    expanded_chunks: List[Chunk] = []

    if (
        include_previous
        and center_chunk.previous_chunk_id is not None
    ):
        previous_chunk = chunk_lookup.get(
            center_chunk.previous_chunk_id
        )

        if (
            previous_chunk is not None
            and is_valid_neighbour(
                center_chunk,
                previous_chunk,
            )
        ):
            expanded_chunks.append(
                previous_chunk
            )

    expanded_chunks.append(
        center_chunk
    )

    if (
        include_next
        and center_chunk.next_chunk_id is not None
    ):
        next_chunk = chunk_lookup.get(
            center_chunk.next_chunk_id
        )

        if (
            next_chunk is not None
            and is_valid_neighbour(
                center_chunk,
                next_chunk,
            )
        ):
            expanded_chunks.append(
                next_chunk
            )

    return expanded_chunks


def expand_context(
    results: List[SearchResult],
    chunks: List[Chunk],
) -> List[Chunk]:
    """
    Expand reranked results with neighbouring chunks
    while avoiding duplicates.

    Final chunks are returned in document order.
    """

    if not results:
        return []

    chunk_lookup = build_chunk_lookup(
        chunks
    )

    selected_chunks: Dict[int, Chunk] = {}

    for result in results:

        expanded = expand_single_result(
            result=result,
            chunk_lookup=chunk_lookup,
        )

        for chunk in expanded:
            selected_chunks[
                chunk.chunk_id
            ] = chunk

    final_chunks = list(
        selected_chunks.values()
    )

    final_chunks.sort(
        key=lambda chunk: chunk.chunk_id
    )

    return final_chunks


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

    from src.reranker import (
        load_reranker,
        rerank_results,
    )

    index, chunks = (
        load_vector_store()
    )

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

    expanded_chunks = expand_context(
        results=reranked_results,
        chunks=chunks,
    )

    print(
        f"Reranked results: "
        f"{len(reranked_results)}"
    )

    print(
        f"Expanded context chunks: "
        f"{len(expanded_chunks)}"
    )

    print()

    for chunk in expanded_chunks:

        print(
            f"Chunk {chunk.chunk_id}"
        )

        print(
            f"Document: "
            f"{chunk.document_name}"
        )

        print(
            f"Page: {chunk.page}"
        )

        print(
            f"Section: {chunk.section}"
        )

        print(
            f"Text: {chunk.text[:300]}"
        )

        print("-" * 80)