from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from models import Chunk


DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"


def load_embedding_model(
    model_name: str = DEFAULT_MODEL_NAME,
) -> SentenceTransformer:
    """
    Load a pretrained Sentence Transformer model.
    """

    return SentenceTransformer(model_name)


def embed_texts(
    texts: List[str],
    model: SentenceTransformer,
) -> np.ndarray:
    """
    Convert texts into normalized embedding vectors.
    """

    if not texts:
        dimension = model.get_sentence_embedding_dimension()

        if dimension is None:
            raise ValueError(
                "Could not determine the embedding dimension."
            )

        return np.empty((0, dimension), dtype=np.float32)

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    return embeddings.astype(np.float32)


if __name__ == "__main__":
    from chunking import chunk_pages
    from ingest import load_pdf

    pages = load_pdf("data/thesis.pdf")

    chunks: List[Chunk] = chunk_pages(
        pages=pages,
        chunk_size=300,
        overlap=50,
    )

    texts = [chunk.text for chunk in chunks]

    model = load_embedding_model()

    embeddings = embed_texts(
        texts=texts,
        model=model,
    )

    print(f"Number of chunks: {len(chunks)}")
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Embedding dtype: {embeddings.dtype}")
    print()
    print("First 10 values of the first embedding:")
    print(embeddings[0][:10])