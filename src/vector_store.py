from pathlib import Path
from typing import List
import json

import faiss
import numpy as np


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """
    Build a FAISS index using inner product similarity.

    The embeddings must already be normalized.
    """

    if embeddings.ndim != 2:
        raise ValueError("embeddings must be a 2D matrix")
    if len(embeddings) == 0:
        raise ValueError("embeddings cannot be empty")

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    return index


def save_vector_store(
    index: faiss.Index,
    chunks: List[dict],
    output_dir: str = "vector_store",
) -> None:
    """
    Save the FAISS index and chunk metadata to disk.
    """

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    faiss.write_index(
        index,
        str(output_path / "index.faiss"),
    )

    with open(
        output_path / "metadata.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            chunks,
            file,
            ensure_ascii=False,
            indent=2,
        )


def load_vector_store(
    output_dir: str = "vector_store",
) -> tuple[faiss.Index, List[dict]]:
    """
    Load the FAISS index and metadata from disk.
    """

    output_path = Path(output_dir)
    index_path = output_path / "index.faiss"
    metadata_path = output_path / "metadata.json"

    if not index_path.exists():
        raise FileNotFoundError(f"{index_path} not found")
    if not metadata_path.exists():
        raise FileNotFoundError(f"{metadata_path} not found")

    index = faiss.read_index(str(index_path))

    with open(metadata_path, "r", encoding="utf-8") as file:
        chunks = json.load(file)

    return index, chunks


if __name__ == "__main__":
    from ingest import load_pdf
    from chunking import chunk_pages
    from embeddings import load_embedding_model, embed_texts

    pages = load_pdf("data/thesis.pdf")
    chunks = chunk_pages(
        pages=pages,
        chunk_size=300,
        overlap=50,
    )
    texts = [chunk["text"] for chunk in chunks]
    model = load_embedding_model()
    embeddings = embed_texts(texts, model)
    index = build_faiss_index(embeddings)

    save_vector_store(
        index=index,
        chunks=chunks,
    )
    print(f"Chunks: {len(chunks)}")
    print(f"Vectors stored in FAISS: {index.ntotal}")
    print(f"Embedding dimension: {index.d}")
    print("Vector store saved successfully.")