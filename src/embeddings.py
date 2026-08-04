from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer


DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"


def load_embedding_model(
    model_name: str = DEFAULT_MODEL_NAME,
) -> SentenceTransformer:
    """
    Load a pretrained Sentence Transformer embedding model.
    Parameters
    ----------
    model_name:
        Name or local path of the embedding model.
    Returns
    -------
    SentenceTransformer
        Loaded embedding model.
    """
    return SentenceTransformer(model_name)


def embed_texts(
    texts: List[str],
    model: SentenceTransformer,
) -> np.ndarray:
    """
    Convert a list of texts into normalized embedding vectors.

    Parameters
    ----------
    texts:
        Texts to encode.
    model:
        Loaded Sentence Transformer model.

    Returns
    -------
    np.ndarray
        Matrix with shape:
        number_of_texts × embedding_dimension
    """

    if not texts:
        return np.empty((0, model.get_sentence_embedding_dimension()))

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    return embeddings.astype("float32")


if __name__ == "__main__":
    from ingest import load_pdf
    from chunking import chunk_pages

    pages = load_pdf("data/thesis.pdf")

    chunks = chunk_pages(
        pages=pages,
        chunk_size=300,
        overlap=50,
    )

    texts = [chunk["text"] for chunk in chunks]

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