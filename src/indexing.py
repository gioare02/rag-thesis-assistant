from src.ingest import load_knowledge_base
from src.chunking import chunk_pages
from src.embeddings import (
    load_embedding_model,
    embed_chunks,
)
from src.vector_store import (
    build_faiss_index,
    save_vector_store,
)


def rebuild_index(
    thesis_directory: str = "data/base",
    papers_directory: str = "data/uploads",
    output_directory: str = "vector_store",
) -> None:
    """
    Rebuild the complete knowledge-base index.

    Pipeline:
        PDFs
        -> pages
        -> structure-aware chunks
        -> embeddings
        -> FAISS
        -> persisted metadata
    """

    print("Loading documents...")

    pages = load_knowledge_base(
        thesis_directory=thesis_directory,
        papers_directory=papers_directory,
    )

    print(
        f"Loaded {len(pages)} pages."
    )

    print("Chunking documents...")

    chunks = chunk_pages(
        pages=pages,
    )

    print(
        f"Created {len(chunks)} chunks."
    )

    print("Loading embedding model...")

    embedding_model = (
        load_embedding_model()
    )

    print("Creating embeddings...")

    embeddings = embed_chunks(
        chunks=chunks,
        model=embedding_model,
    )

    print(
        f"Embeddings shape: "
        f"{embeddings.shape}"
    )

    print("Building FAISS index...")

    index = build_faiss_index(
        embeddings
    )

    print(
        f"FAISS vectors: "
        f"{index.ntotal}"
    )

    print("Saving vector store...")

    save_vector_store(
        index=index,
        chunks=chunks,
        output_dir=output_directory,
    )

    print()
    print("Index rebuild completed.")


if __name__ == "__main__":

    rebuild_index()