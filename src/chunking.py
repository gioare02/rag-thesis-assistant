from typing import List
from src.models import Page, Chunk
from src.config import CHUNK_SIZE, CHUNK_OVERLAP

def split_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Split text into overlapping word-based chunks.
    Parameters
    ----------
    text:
        Text to split.
    chunk_size:
        Maximum number of words in each chunk.
    overlap:
        Number of words shared between consecutive chunks.

    Returns
    -------
    List[str]
        List of text chunks.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if overlap < 0:
        raise ValueError("overlap cannot be negative")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    
    words = text.split()

    if not words:
        return []

    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunk_text = " ".join(chunk_words)

        chunks.append(chunk_text)
        start += chunk_size - overlap

    return chunks

def chunk_pages(pages: List[Page], chunk_size: int = 300, overlap: int = 50) -> List[Chunk]:
    """
    Split all pages into chunks while preserving metadata.
    """

    chunks = []
    chunk_id = 0

    for page in pages:
        page_chunks = split_text(
            text=page.text,
            chunk_size=chunk_size,
            overlap=overlap,
        )
        for chunk_text in page_chunks:
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    document=page.document,
                    page=page.page,
                    text=chunk_text,
                )
            )
            chunk_id += 1

    return chunks




######################### TEST ##########################

if __name__ == "__main__":
    from ingest import load_pdf
    pages = load_pdf("data/thesis.pdf")
    chunks = chunk_pages(
        pages=pages,
        chunk_size=CHUNK_SIZE,
        overlap=CHUNK_OVERLAP,
    )
    print(f"Pages: {len(pages)}")
    print(f"Chunks: {len(chunks)}")
    print()
    print(chunks[0])