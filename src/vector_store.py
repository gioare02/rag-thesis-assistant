from dataclasses import asdict
from pathlib import Path
from typing import List
import json

import faiss
import numpy as np

from src.models import Chunk


def build_faiss_index(
    embeddings: np.ndarray,
) -> faiss.Index:
    """
    Build an exact FAISS index using inner-product similarity.

    Because embeddings are normalized, inner product
    corresponds to cosine similarity.
    """

    if embeddings.ndim != 2:
        raise ValueError(
            "embeddings must be a 2D matrix"
        )

    if len(embeddings) == 0:
        raise ValueError(
            "embeddings cannot be empty"
        )

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(
        dimension
    )

    index.add(embeddings)

    return index


def save_vector_store(
    index: faiss.Index,
    chunks: List[Chunk],
    output_dir: str = "vector_store",
) -> None:
    """
    Persist the FAISS index and chunk metadata to disk.
    """

    output_path = Path(output_dir)

    output_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    faiss.write_index(
        index,
        str(output_path / "index.faiss"),
    )

    chunk_data = [
        asdict(chunk)
        for chunk in chunks
    ]

    with open(
        output_path / "metadata.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            chunk_data,
            file,
            ensure_ascii=False,
            indent=2,
        )


def load_vector_store(
    output_dir: str = "vector_store",
) -> tuple[faiss.Index, List[Chunk]]:
    """
    Load the FAISS index and reconstruct Chunk objects
    from the stored metadata.
    """

    output_path = Path(output_dir)

    index_path = (
        output_path / "index.faiss"
    )

    metadata_path = (
        output_path / "metadata.json"
    )

    if not index_path.exists():
        raise FileNotFoundError(
            f"{index_path} not found"
        )

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"{metadata_path} not found"
        )

    index = faiss.read_index(
        str(index_path)
    )

    with open(
        metadata_path,
        "r",
        encoding="utf-8",
    ) as file:

        chunk_data = json.load(file)

    chunks = [
        Chunk(**item)
        for item in chunk_data
    ]

    return index, chunks


if __name__ == "__main__":

    from src.ingest import load_knowledge_base
    from src.chunking import chunk_pages
    from src.embeddings import (
        load_embedding_model,
        embed_chunks,
    )

    pages = load_knowledge_base(
        thesis_directory="data/base",
        papers_directory="data/uploads",
    )

    chunks = chunk_pages(pages)

    print(f"Pages: {len(pages)}")
    print(f"Chunks: {len(chunks)}")

    model = load_embedding_model()

    embeddings = embed_chunks(
        chunks=chunks,
        model=model,
    )

    print(
        f"Embeddings shape: "
        f"{embeddings.shape}"
    )

    index = build_faiss_index(
        embeddings
    )

    print(
        f"Vectors in FAISS index: "
        f"{index.ntotal}"
    )

    save_vector_store(
        index=index,
        chunks=chunks,
    )

    loaded_index, loaded_chunks = (
        load_vector_store()
    )

    print()
    print("Reload test")
    print(
        f"Loaded vectors: "
        f"{loaded_index.ntotal}"
    )
    print(
        f"Loaded chunks: "
        f"{len(loaded_chunks)}"
    )

    if loaded_chunks:

        first_chunk = loaded_chunks[0]

        print()
        print("First loaded chunk:")
        print(
            f"ID: {first_chunk.chunk_id}"
        )
        print(
            f"Document: "
            f"{first_chunk.document_name}"
        )
        print(
            f"Type: "
            f"{first_chunk.document_type}"
        )
        print(
            f"Page: "
            f"{first_chunk.page}"
        )
        print(
            f"Section: "
            f"{first_chunk.section}"
        )