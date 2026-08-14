from typing import List, Dict

from src.models import Page, Chunk
from src.config import CHUNK_SIZE, CHUNK_OVERLAP


def split_words(
    words: List[str],
    chunk_size: int,
    overlap: int,
) -> List[List[str]]:
    """
    Split a list of words into overlapping chunks.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")

    if overlap < 0:
        raise ValueError("overlap cannot be negative")

    if overlap >= chunk_size:
        raise ValueError(
            "overlap must be smaller than chunk_size"
        )

    if not words:
        return []

    chunks = []

    start = 0

    while start < len(words):

        end = start + chunk_size

        chunk_words = words[start:end]

        chunks.append(chunk_words)

        start += chunk_size - overlap

    return chunks


def group_pages_by_section(
    pages: List[Page],
) -> List[List[Page]]:
    """
    Group consecutive pages that belong to the same
    document and section.

    Example:

        Page 10 -> Results
        Page 11 -> Results
        Page 12 -> Discussion

    becomes:

        [
            [Page 10, Page 11],
            [Page 12]
        ]
    """

    if not pages:
        return []

    groups: List[List[Page]] = []

    current_group = [pages[0]]

    for page in pages[1:]:

        previous_page = current_group[-1]

        same_document = (
            page.document_id
            == previous_page.document_id
        )

        same_section = (
            page.section
            == previous_page.section
        )

        if same_document and same_section:

            current_group.append(page)

        else:

            groups.append(current_group)

            current_group = [page]

    groups.append(current_group)

    return groups


def build_section_words(
    section_pages: List[Page],
) -> List[Dict]:
    """
    Convert all words in a section into a sequence
    that remembers which page each word came from.

    Example:

        [
            {"word": "sentiment", "page": 40},
            {"word": "improves", "page": 40},
            ...
            {"word": "volatility", "page": 41}
        ]
    """

    words_with_metadata = []

    for page in section_pages:

        words = page.text.split()

        for word in words:

            words_with_metadata.append(
                {
                    "word": word,
                    "page": page.page,
                }
            )

    return words_with_metadata


def chunk_section(
    section_pages: List[Page],
    start_chunk_id: int,
    chunk_size: int,
    overlap: int,
) -> List[Chunk]:
    """
    Chunk a complete logical section rather than
    processing each PDF page independently.
    """

    if not section_pages:
        return []

    first_page = section_pages[0]

    words_with_metadata = build_section_words(
        section_pages
    )

    if not words_with_metadata:
        return []

    chunks: List[Chunk] = []

    start = 0
    chunk_id = start_chunk_id

    while start < len(words_with_metadata):

        end = start + chunk_size

        chunk_items = words_with_metadata[start:end]

        if not chunk_items:
            break

        chunk_text = " ".join(
            item["word"]
            for item in chunk_items
        )

        start_page = chunk_items[0]["page"]

        chunk = Chunk(
            chunk_id=chunk_id,

            document_id=first_page.document_id,
            document_name=first_page.document_name,
            document_type=first_page.document_type,

            page=start_page,
            section=first_page.section,

            text=chunk_text,
        )

        chunks.append(chunk)

        chunk_id += 1

        start += chunk_size - overlap

    return chunks


def chunk_pages(
    pages: List[Page],
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> List[Chunk]:
    """
    Structure-aware chunking pipeline.

    1. Group consecutive pages by section.
    2. Combine text inside each logical section.
    3. Split each section into overlapping chunks.
    4. Preserve source metadata.
    5. Add previous/next chunk relationships.
    """

    section_groups = group_pages_by_section(pages)

    chunks: List[Chunk] = []

    next_chunk_id = 0

    for section_pages in section_groups:

        section_chunks = chunk_section(
            section_pages=section_pages,
            start_chunk_id=next_chunk_id,
            chunk_size=chunk_size,
            overlap=overlap,
        )

        chunks.extend(section_chunks)

        next_chunk_id += len(section_chunks)

    add_chunk_links(chunks)

    return chunks


def add_chunk_links(
    chunks: List[Chunk],
) -> None:
    """
    Add previous/next relationships between chunks.

    Links are created only between chunks belonging
    to the same document.
    """

    for i, chunk in enumerate(chunks):

        if i > 0:

            previous_chunk = chunks[i - 1]

            if (
                previous_chunk.document_id
                == chunk.document_id
            ):
                chunk.previous_chunk_id = (
                    previous_chunk.chunk_id
                )

        if i < len(chunks) - 1:

            next_chunk = chunks[i + 1]

            if (
                next_chunk.document_id
                == chunk.document_id
            ):
                chunk.next_chunk_id = (
                    next_chunk.chunk_id
                )


if __name__ == "__main__":

    from src.ingest import load_knowledge_base

    pages = load_knowledge_base(
        thesis_directory="data/base",
        papers_directory="data/uploads",
    )

    chunks = chunk_pages(
        pages=pages,
        chunk_size=CHUNK_SIZE,
        overlap=CHUNK_OVERLAP,
    )

    print(f"Pages: {len(pages)}")
    print(f"Chunks: {len(chunks)}")
    print()

    for chunk in chunks[:10]:

        print(
            f"Chunk {chunk.chunk_id}"
        )

        print(
            f"Document: {chunk.document_name}"
        )

        print(
            f"Type: {chunk.document_type}"
        )

        print(
            f"Page: {chunk.page}"
        )

        print(
            f"Section: {chunk.section}"
        )

        print(
            f"Previous: {chunk.previous_chunk_id}"
        )

        print(
            f"Next: {chunk.next_chunk_id}"
        )

        print(
            f"Text: {chunk.text[:300]}..."
        )

        print("-" * 80)