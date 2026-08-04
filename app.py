from pathlib import Path

import streamlit as st

from src.bm25_retriever import build_bm25_retriever
from src.config import (
    HYBRID_CANDIDATES,
    RERANK_TOP_K,
    TOP_K,
)
from src.reranker import load_reranker, rerank_results
from src.embeddings import load_embedding_model
from src.hybrid_retriever import fuse_results_rrf
from src.indexing import build_vector_store
from src.langchain_documents import chunks_to_documents
from src.llm import create_client, generate_answer, rewrite_query
from src.retriever import retrieve_chunks
from src.vector_store import load_vector_store


BASE_DIR = Path("data/base")
UPLOAD_DIR = Path("data/uploads")

BASE_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


st.set_page_config(
    page_title="ThesisRAG",
    page_icon="📚",
    layout="wide",
)


@st.cache_resource
def load_resources():
    embedding_model = load_embedding_model()
    reranker = load_reranker()

    index, chunks = load_vector_store()
    client = create_client()

    documents = chunks_to_documents(chunks)

    bm25_retriever = build_bm25_retriever(
        documents=documents,
        top_k=HYBRID_CANDIDATES,
    )

    return (
        embedding_model,
        reranker,
        index,
        chunks,
        client,
        bm25_retriever,
    )


def rebuild_vector_store(embedding_model) -> None:
    """
    Rebuild the vector store using permanent and uploaded documents.
    """

    build_vector_store(
        base_directory=str(BASE_DIR),
        upload_directory=str(UPLOAD_DIR),
        embedding_model=embedding_model,
    )

    # Clear cached resources because the FAISS index
    # and the BM25 retriever must be reloaded.
    load_resources.clear()

    st.session_state.messages = []


st.title("📚 ThesisRAG")

st.caption(
    "Ask questions about the thesis and uploaded research papers. "
    "Answers are generated only from retrieved document context."
)


if "messages" not in st.session_state:
    st.session_state.messages = []


try:
    (
        embedding_model,
        reranker,
        index,
        chunks,
        client,
        bm25_retriever,
    ) = load_resources()

except Exception as error:
    st.error(f"Could not load the RAG system: {error}")

    st.info(
        "Make sure the vector store has been created and "
        "the OpenAI API key is available in the .env file."
    )

    st.stop()


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:
    st.header("Documents")

    # Permanent documents
    st.subheader("Permanent knowledge base")

    base_documents = sorted(BASE_DIR.glob("*.pdf"))

    if not base_documents:
        st.warning("No permanent PDF found in data/base.")
    else:
        for document in base_documents:
            st.write(f"📄 {document.name}")

    st.divider()

    # Upload new papers
    st.subheader("Upload papers")

    uploaded_files = st.file_uploader(
        "Upload additional PDF documents",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        if st.button(
            "Save and index documents",
            type="primary",
            use_container_width=True,
        ):
            with st.spinner("Saving and indexing documents..."):
                for uploaded_file in uploaded_files:
                    destination = UPLOAD_DIR / uploaded_file.name

                    with open(destination, "wb") as file:
                        file.write(uploaded_file.getbuffer())

                rebuild_vector_store(embedding_model)

            st.success("Knowledge base indexed successfully.")
            st.rerun()

    st.divider()

    # Existing uploaded papers
    st.subheader("Uploaded papers")

    uploaded_documents = sorted(UPLOAD_DIR.glob("*.pdf"))

    if not uploaded_documents:
        st.caption("No uploaded papers.")

    else:
        for document in uploaded_documents:
            document_column, delete_column = st.columns([4, 1])

            with document_column:
                st.write(f"📄 {document.name}")

            with delete_column:
                if st.button(
                    "🗑️",
                    key=f"delete_{document.name}",
                    help=f"Delete {document.name}",
                ):
                    with st.spinner(
                        f"Deleting {document.name} "
                        "and rebuilding the index..."
                    ):
                        document.unlink()

                        rebuild_vector_store(embedding_model)

                    st.success(f"{document.name} deleted.")
                    st.rerun()

    st.divider()

    # Manual index rebuild
    if st.button(
        "Rebuild index",
        use_container_width=True,
    ):
        with st.spinner("Rebuilding vector store..."):
            rebuild_vector_store(embedding_model)

        st.success("Vector store rebuilt successfully.")
        st.rerun()

    # Clear chat
    if st.button(
        "Clear conversation",
        use_container_width=True,
    ):
        st.session_state.messages = []
        st.rerun()


# =========================================================
# CHAT HISTORY
# =========================================================

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if (
            message["role"] == "assistant"
            and "rewritten_query" in message
            and message["rewritten_query"] != message.get(
                "original_question"
            )
        ):
            with st.expander("Query rewriting"):
                st.write("Original question:")
                st.code(message["original_question"])

                st.write("Standalone retrieval query:")
                st.code(message["rewritten_query"])

        if (
            message["role"] == "assistant"
            and "sources" in message
        ):
            with st.expander("Retrieved sources"):
                for rank, source in enumerate(
                    message["sources"],
                    start=1,
                ):
                    st.markdown(
                        f"**Source {rank} — "
                        f"{source['document']}, "
                        f"page {source['page']}**"
                    )

                    st.caption(
                        f"Reranker score: {source['score']:.6f}"
                    )

                    st.write(source["text"])
                    st.divider()


# =========================================================
# NEW USER QUESTION
# =========================================================

question = st.chat_input(
    "Ask a question about the thesis or uploaded papers"
)


if question:
    # Keep the previous history separate because query rewriting
    # should use only earlier messages.
    previous_messages = st.session_state.messages.copy()

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching the knowledge base..."):
            # 1. Rewrite conversational question as standalone query
            rewritten_query = rewrite_query(
                question=question,
                chat_history=previous_messages,
                client=client,
            )

            # 2. Semantic retrieval with FAISS
            semantic_results = retrieve_chunks(
                query=rewritten_query,
                model=embedding_model,
                index=index,
                chunks=chunks,
                top_k=HYBRID_CANDIDATES,
            )

            lexical_results = bm25_retriever.invoke(
                rewritten_query
            )

            hybrid_results = fuse_results_rrf(
                semantic_results=semantic_results,
                lexical_results=lexical_results,
                top_k=HYBRID_CANDIDATES,
            )

            results = rerank_results(
                query=rewritten_query,
                results=hybrid_results,
                reranker=reranker,
                top_k=RERANK_TOP_K,
            )

            # 5. Generate grounded answer
            answer = generate_answer(
                question=question,
                results=results,
                client=client,
            )

        st.markdown(answer)

        if rewritten_query != question:
            with st.expander("Query rewriting"):
                st.write("Original question:")
                st.code(question)

                st.write("Standalone retrieval query:")
                st.code(rewritten_query)

        with st.expander("Retrieved sources"):
            for rank, result in enumerate(
                results,
                start=1,
            ):
                st.markdown(
                    f"**Source {rank} — "
                    f"{result.chunk.document}, "
                    f"page {result.chunk.page}**"
                )

                st.caption(
                    f"Reranker score: {result.score:.6f}"
                )

                st.write(result.chunk.text)
                st.divider()

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "original_question": question,
            "rewritten_query": rewritten_query,
            "sources": [
                {
                    "document": result.chunk.document,
                    "page": result.chunk.page,
                    "score": result.score,
                    "text": result.chunk.text,
                }
                for result in results
            ],
        }
    )

'''
Domanda
   ↓
Query rewriting
   ↓
FAISS top 10 + BM25 top 10
   ↓
RRF top 10
   ↓
CrossEncoder reranker
   ↓
Top 5
   ↓
LLM
'''