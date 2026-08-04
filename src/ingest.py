from pathlib import Path
import fitz  # type: ignore # PyMuPDF
from typing import List

def load_pdf(pdf_path: str)  -> List[dict]:
    """
    Read a PDF and return one record per page.
    Returns
    -------
    list[dict]
    """

    if not Path(pdf_path).exists():
        raise FileNotFoundError(f"{pdf_path} not found")

    pdf = fitz.open(pdf_path)
    pages = []

    for page_number in range(len(pdf)):
        page = pdf.load_page(page_number)
        text = page.get_text()
        pages.append(
            {
                "document": Path(pdf_path).name,
                "page": page_number + 1,
                "text": text,
            }
        )
    pdf.close()
    return pages


if __name__ == "__main__":
    pages = load_pdf("data/thesis.pdf")
    print(f"Pages: {len(pages)}")
    print()
    print(pages[0])