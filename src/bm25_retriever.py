from typing import List, Optional

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from src.models import Chunk, SearchResult
from src.langchain_documents import chunks_to_documents
from src.config import TOP_K


def filter_chunks(
    chunks: List[Chunk],
    document_type: Optional[str] = None,
    document_ids: Optional[List[str]] = None,
) -> List[Chunk]:
    """
    Filter chunks using document metadata before
    constructing the BM25 retriever.
    """

    filtered_chunks = []

    for chunk in chunks:

        if document_type is not None:
            if chunk.document_type != document_type:
                continue

        if document_ids is not None:
            if chunk.document_id not in document_ids:
                continue

        filtered_chunks.append(chunk)

    return filtered_chunks


def build_bm25_retriever(
    chunks: List[Chunk],
    top_k: int = TOP_K,
    document_type: Optional[str] = None,
    document_ids: Optional[List[str]] = None,
) -> BM25Retriever:
    """
    Build a BM25 lexical retriever.

    Optional metadata filters are applied before
    building the retriever.
    """

    if top_k <= 0:
        raise ValueError(
            "top_k must be greater than 0"
        )

    filtered_chunks = filter_chunks(
        chunks=chunks,
        document_type=document_type,
        document_ids=document_ids,
    )

    if not filtered_chunks:
        raise ValueError(
            "No chunks available after metadata filtering."
        )

    documents = chunks_to_documents(
        filtered_chunks
    )

    retriever = BM25Retriever.from_documents(
        documents
    )

    retriever.k = min(
        top_k,
        len(documents),
    )

    return retriever


def retrieve_bm25(
    query: str,
    chunks: List[Chunk],
    top_k: int = TOP_K,
    document_type: Optional[str] = None,
    document_ids: Optional[List[str]] = None,
) -> List[SearchResult]:
    """
    Retrieve relevant chunks using BM25 lexical search.
    """

    if not query.strip():
        raise ValueError(
            "query cannot be empty"
        )

    retriever = build_bm25_retriever(
        chunks=chunks,
        top_k=top_k,
        document_type=document_type,
        document_ids=document_ids,
    )

    documents = retriever.invoke(query)

    chunk_lookup = {
        chunk.chunk_id: chunk
        for chunk in chunks
    }

    results = []

    for rank, document in enumerate(
        documents,
        start=1,
    ):

        chunk_id = document.metadata[
            "chunk_id"
        ]

        chunk = chunk_lookup[chunk_id]

        # LangChain's BM25Retriever does not expose
        # the raw BM25 score here, so we use rank-based
        # scores for a common SearchResult interface.
        score = 1.0 / rank

        results.append(
            SearchResult(
                chunk=chunk,
                score=score,
            )
        )

    return results


if __name__ == "__main__":

    from src.vector_store import (
        load_vector_store,
    )

    _, chunks = load_vector_store()

    query = (
        "CONF 0.90 KNN agreement filtering"
    )

    results = retrieve_bm25(
        query=query,
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
            f"Rank score: {result.score:.4f}"
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