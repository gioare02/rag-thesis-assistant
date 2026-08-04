from pathlib import Path
from typing import List

import fitz

from src.models import Page


def load_pdf(pdf_path: str) -> List[Page]:
    path = Path(pdf_path)

    if not path.exists():
        raise FileNotFoundError(f"{pdf_path} not found")

    pages: List[Page] = []

    with fitz.open(path) as pdf:
        for page_number in range(len(pdf)):
            page = pdf.load_page(page_number)

            pages.append(
                Page(
                    document=path.name,
                    page=page_number + 1,
                    text=page.get_text(),
                )
            )

    return pages


def load_pdfs_from_directory(directory: str) -> List[Page]:
    directory_path = Path(directory)

    if not directory_path.exists():
        return []

    pages: List[Page] = []

    for pdf_path in sorted(directory_path.glob("*.pdf")):
        pages.extend(load_pdf(str(pdf_path)))

    return pages


def load_knowledge_base(
    base_directory: str,
    upload_directory: str,
) -> List[Page]:
    """
    Load permanent documents and user-uploaded documents.
    """

    base_pages = load_pdfs_from_directory(base_directory)
    uploaded_pages = load_pdfs_from_directory(upload_directory)

    pages = base_pages + uploaded_pages

    if not pages:
        raise ValueError("No PDF documents found in the knowledge base.")

    return pages




######################### TEST ##########################

if __name__ == "__main__":
    pages = load_pdf("data/thesis.pdf")
    print(f"Pages: {len(pages)}")
    print()
    print(pages[0])