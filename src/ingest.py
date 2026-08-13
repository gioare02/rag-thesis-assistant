from pathlib import Path
from typing import List
import re

import fitz

from src.models import Page


def make_document_id(filename: str) -> str:
    """
    Convert a filename into a simple internal document id.

    Example:
        "My Paper 2024.pdf" -> "my_paper_2024"
    """

    stem = Path(filename).stem.lower()

    document_id = re.sub(r"[^a-z0-9]+", "_", stem)

    return document_id.strip("_")


def load_pdf(
    pdf_path: str,
    document_type: str,
    document_id: str | None = None,
) -> List[Page]:
    """
    Load a PDF and return one Page object for each PDF page.
    """

    path = Path(pdf_path)

    if not path.exists():
        raise FileNotFoundError(f"{pdf_path} not found")

    if document_id is None:
        document_id = make_document_id(path.name)

    pages: List[Page] = []

    with fitz.open(path) as pdf:

        for page_number in range(len(pdf)):

            page = pdf.load_page(page_number)

            text = page.get_text().strip()

            pages.append(
                Page(
                    document_id=document_id,
                    document_name=path.name,
                    document_type=document_type,
                    page=page_number + 1,
                    text=text,
                )
            )

    return pages


def load_pdfs_from_directory(
    directory: str,
    document_type: str,
) -> List[Page]:
    """
    Load all PDF documents from a directory.
    """

    directory_path = Path(directory)

    if not directory_path.exists():
        return []

    pages: List[Page] = []

    for pdf_path in sorted(directory_path.glob("*.pdf")):

        pdf_pages = load_pdf(
            pdf_path=str(pdf_path),
            document_type=document_type,
        )

        pages.extend(pdf_pages)

    return pages


def load_knowledge_base(
    thesis_directory: str,
    papers_directory: str,
) -> List[Page]:
    """
    Load the thesis and uploaded research papers.
    """

    thesis_pages = load_pdfs_from_directory(
        directory=thesis_directory,
        document_type="thesis",
    )

    paper_pages = load_pdfs_from_directory(
        directory=papers_directory,
        document_type="paper",
    )

    pages = thesis_pages + paper_pages

    if not pages:
        raise ValueError(
            "No PDF documents found in the knowledge base."
        )

    return pages


######################### TEST ##########################

if __name__ == "__main__":

    pages = load_knowledge_base(
        thesis_directory="data/base",
        papers_directory="data/uploads",
    )

    print(f"Total pages: {len(pages)}")
    print()

    for page in pages[:3]:
        print(page)
        print()


