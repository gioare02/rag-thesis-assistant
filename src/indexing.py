from typing import List

from src.chunking import chunk_pages
from src.config import CHUNK_OVERLAP, CHUNK_SIZE
from src.embeddings import embed_texts
from src.ingest import load_knowledge_base
from src.models import Chunk
from src.vector_store import build_faiss_index, save_vector_store


def build_vector_store(
    base_directory: str,
    upload_directory: str,
    embedding_model,
    output_dir: str = "vector_store",
) -> List[Chunk]:
    """
    Build and save a FAISS vector store using both permanent
    and uploaded PDF documents.
    """

    pages = load_knowledge_base(
        base_directory=base_directory,
        upload_directory=upload_directory,
    )

    chunks = chunk_pages(
        pages=pages,
        chunk_size=CHUNK_SIZE,
        overlap=CHUNK_OVERLAP,
    )

    texts = [chunk.text for chunk in chunks]

    embeddings = embed_texts(
        texts=texts,
        model=embedding_model,
    )

    index = build_faiss_index(embeddings)

    save_vector_store(
        index=index,
        chunks=chunks,
        output_dir=output_dir,
    )

    return chunks