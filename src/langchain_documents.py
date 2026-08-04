from typing import List
from langchain_core.documents import Document
from src.models import Chunk

def chunks_to_documents(chunks: List[Chunk]) -> List[Document]:
    """
    Convert internal Chunk objects into LangChain Documents.
    """
    return [
        Document(
            page_content=chunk.text,
            metadata={
                "chunk_id": chunk.chunk_id,
                "document": chunk.document,
                "page": chunk.page,
            },
        )
        for chunk in chunks
    ]