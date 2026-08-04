from typing import List, Dict
from collections import defaultdict

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from src.models import SearchResult

def build_bm25_retriever(
    documents: List[Document],
    top_k: int = 10,
) -> BM25Retriever:
    """
    Build a BM25 lexical retriever.
    """

    if not documents:
        raise ValueError("documents cannot be empty")

    retriever = BM25Retriever.from_documents(documents)
    retriever.k = top_k

    return retriever



######################### TEST ##########################

if __name__ == "__main__":
    from src.langchain_documents import chunks_to_documents
    from src.vector_store import load_vector_store

    _, chunks = load_vector_store()
    documents = chunks_to_documents(chunks)
    retriever = build_bm25_retriever(
        documents=documents,
        top_k=5,
    )

    query = "CONF 0.97 KNN 0.90"
    results = retriever.invoke(query)

    print(f"Query: {query}")
    print()

    for rank, document in enumerate(results, start=1):
        print(f"Result {rank}")
        print(f"Document: {document.metadata['document']}")
        print(f"Page: {document.metadata['page']}")
        print(document.page_content[:500])
        print("-" * 80)