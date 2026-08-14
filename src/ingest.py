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


def extract_section_title(page) -> str | None:
    """
    Try to detect the main section title on a PDF page
    using layout/font-size information from PyMuPDF.

    This is a lightweight heuristic:
    - short text is more likely to be a heading
    - larger font is more likely to be a heading
    """

    page_dict = page.get_text("dict")

    candidates = []

    for block in page_dict.get("blocks", []):

        if "lines" not in block:
            continue

        for line in block["lines"]:

            for span in line.get("spans", []):

                text = span.get("text", "").strip()
                font_size = span.get("size", 0)

                if not text:
                    continue

                # Headings are usually relatively short
                if len(text.split()) > 12:
                    continue

                # Ignore page numbers such as "42"
                if text.isdigit():
                    continue

                # Ignore common figure/table labels
                lower_text = text.lower()

                if lower_text.startswith(("figure ", "table ")):
                    continue

                candidates.append(
                    {
                        "text": text,
                        "size": font_size,
                    }
                )

    if not candidates:
        return None

    max_font_size = max(
        candidate["size"]
        for candidate in candidates
    )

    largest_texts = [
        candidate["text"]
        for candidate in candidates
        if candidate["size"] == max_font_size
    ]

    if not largest_texts:
        return None

    return " ".join(largest_texts)


def load_pdf(
    pdf_path: str,
    document_type: str,
    document_id: str | None = None,
) -> List[Page]:
    """
    Load a PDF and return one Page object for each PDF page.

    Each page keeps:
    - document id
    - document name
    - document type
    - page number
    - extracted text
    - detected section
    """

    path = Path(pdf_path)

    if not path.exists():
        raise FileNotFoundError(f"{pdf_path} not found")

    if document_id is None:
        document_id = make_document_id(path.name)

    pages: List[Page] = []

    with fitz.open(path) as pdf:

        current_section = None

        for page_number in range(len(pdf)):

            page = pdf.load_page(page_number)

            text = page.get_text().strip()

            detected_section = extract_section_title(page)

            if detected_section is not None:
                current_section = detected_section

            pages.append(
                Page(
                    document_id=document_id,
                    document_name=path.name,
                    document_type=document_type,
                    page=page_number + 1,
                    text=text,
                    section=current_section,
                )
            )

    return pages


def load_pdfs_from_directory(
    directory: str,
    document_type: str,
) -> List[Page]:
    """
    Load all PDFs contained in a directory.
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
    Load the thesis and all uploaded research papers.
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


if __name__ == "__main__":

    pages = load_knowledge_base(
        thesis_directory="data/base",
        papers_directory="data/uploads",
    )

    print(f"Total pages: {len(pages)}")
    print()

    for page in pages:

        print(
            f"Page {page.page:3} | "
            f"{page.document_name:20} | "
            f"Type: {page.document_type:7} | "
            f"Section: {page.section}"
        )


