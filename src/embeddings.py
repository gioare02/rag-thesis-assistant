from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from src.models import Chunk
from src.config import EMBEDDING_MODEL_NAME


def load_embedding_model(
    model_name: str = EMBEDDING_MODEL_NAME,
) -> SentenceTransformer:
    """
    Load the Sentence Transformer used for dense retrieval.
    """

    return SentenceTransformer(model_name)


def embed_texts(
    texts: List[str],
    model: SentenceTransformer,
) -> np.ndarray:
    """
    Convert a list of texts into normalized dense embeddings.

    Normalized embeddings allow inner-product similarity
    to behave like cosine similarity.
    """

    if not texts:

        dimension = model.get_sentence_embedding_dimension()

        if dimension is None:
            raise ValueError(
                "Could not determine the embedding dimension."
            )

        return np.empty(
            (0, dimension),
            dtype=np.float32,
        )

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    return embeddings.astype(np.float32)


def embed_chunks(
    chunks: List[Chunk],
    model: SentenceTransformer,
) -> np.ndarray:
    """
    Embed the text content of a list of Chunk objects.
    """

    texts = [
        chunk.text
        for chunk in chunks
    ]

    return embed_texts(
        texts=texts,
        model=model,
    )


if __name__ == "__main__":

    from src.ingest import load_knowledge_base
    from src.chunking import chunk_pages

    pages = load_knowledge_base(
        thesis_directory="data/base",
        papers_directory="data/uploads",
    )

    chunks = chunk_pages(pages)

    model = load_embedding_model()

    embeddings = embed_chunks(
        chunks=chunks,
        model=model,
    )

    print(f"Chunks: {len(chunks)}")
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Embedding dtype: {embeddings.dtype}")

    if len(embeddings) > 0:

        print()
        print("First 10 values of first embedding:")
        print(embeddings[0][:10])