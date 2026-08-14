from typing import List

from langchain_core.documents import Document

from src.models import Chunk


def chunk_to_document(
    chunk: Chunk,
) -> Document:
    """
    Convert one internal Chunk object into
    a LangChain Document.
    """

    return Document(
        page_content=chunk.text,
        metadata={
            "chunk_id": chunk.chunk_id,
            "document_id": chunk.document_id,
            "document_name": chunk.document_name,
            "document_type": chunk.document_type,
            "page": chunk.page,
            "section": chunk.section,
            "previous_chunk_id": chunk.previous_chunk_id,
            "next_chunk_id": chunk.next_chunk_id,
        },
    )


def chunks_to_documents(
    chunks: List[Chunk],
) -> List[Document]:
    """
    Convert a list of Chunk objects into
    LangChain Documents.
    """

    return [
        chunk_to_document(chunk)
        for chunk in chunks
    ]