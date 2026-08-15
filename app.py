from pathlib import Path

import streamlit as st

from src.llm import create_client
from src.embeddings import (
    load_embedding_model,
)
from src.reranker import (
    load_reranker,
)
from src.vector_store import (
    load_vector_store,
)
from src.rag_pipeline import (
    RAGPipeline,
)
from src.indexing import (
    rebuild_index,
)


# =========================================================
# Configuration
# =========================================================

UPLOAD_DIRECTORY = Path(
    "data/uploads"
)

UPLOAD_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True,
)


st.set_page_config(
    page_title="Research Copilot",
    page_icon="📚",
    layout="wide",
)


# =========================================================
# Resource loading
# =========================================================

@st.cache_resource
def load_pipeline():
    """
    Load expensive resources only once.

    Streamlit will reuse:
    - embedding model
    - CrossEncoder
    - FAISS index
    - chunks
    - OpenAI client
    """

    client = create_client()

    embedding_model = (
        load_embedding_model()
    )

    reranker = (
        load_reranker()
    )

    index, chunks = (
        load_vector_store()
    )

    return RAGPipeline(
        client=client,
        embedding_model=embedding_model,
        reranker=reranker,
        index=index,
        chunks=chunks,
    )


# =========================================================
# Sidebar
# =========================================================

st.sidebar.title(
    "Knowledge Base"
)

st.sidebar.markdown(
    """
The assistant can reason over:

- your thesis
- uploaded research papers
"""
)


uploaded_files = (
    st.sidebar.file_uploader(
        "Upload research papers",
        type=["pdf"],
        accept_multiple_files=True,
    )
)


if uploaded_files:

    if st.sidebar.button(
        "Add papers and rebuild index"
    ):

        saved_files = []

        for uploaded_file in uploaded_files:

            safe_name = Path(
                uploaded_file.name
            ).name

            destination = (
                UPLOAD_DIRECTORY
                / safe_name
            )

            with open(
                destination,
                "wb",
            ) as file:

                file.write(
                    uploaded_file.getbuffer()
                )

            saved_files.append(
                safe_name
            )

        with st.spinner(
            "Rebuilding the knowledge base..."
        ):

            rebuild_index()

        # The cached pipeline still contains
        # the old FAISS index, so clear it.
        load_pipeline.clear()

        st.sidebar.success(
            f"Added {len(saved_files)} paper(s)."
        )

        st.rerun()


# =========================================================
# Current documents
# =========================================================

pipeline = load_pipeline()


documents = {}

for chunk in pipeline.chunks:

    documents[
        chunk.document_id
    ] = (
        chunk.document_name,
        chunk.document_type,
    )


with st.sidebar.expander(
    "Available documents"
):

    for (
        document_id,
        (
            document_name,
            document_type,
        ),
    ) in documents.items():

        st.write(
            f"**{document_name}** "
            f"— {document_type}"
        )


# =========================================================
# Main UI
# =========================================================

st.title(
    "Research Copilot"
)

st.caption(
    "Ask questions, compare your thesis with "
    "research papers, or validate claims "
    "against the uploaded literature."
)


# =========================================================
# Conversation state
# =========================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# =========================================================
# User question
# =========================================================

question = st.chat_input(
    "Ask about your thesis or research papers..."
)


if question:

    # Show user message

    with st.chat_message("user"):

        st.markdown(question)

    # Previous conversation is passed
    # to query rewriting.
    chat_history = (
        st.session_state.messages.copy()
    )

    with st.spinner(
        "Searching the research corpus..."
    ):

        response = pipeline.run(
            question=question,
            chat_history=chat_history,
        )

    answer = response["answer"]

    route = response["route"]

    # Save conversation

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
        }
    )

    # =====================================================
    # Answer
    # =====================================================

    with st.chat_message(
        "assistant"
    ):

        st.markdown(answer)

        # ---------------------------------
        # Route / reasoning
        # ---------------------------------

        with st.expander(
            "How the query was processed"
        ):

            st.write(
                f"**Detected intent:** "
                f"{route.intent}"
            )

            st.write(
                f"**Document scope:** "
                f"{route.document_type or 'thesis + papers'}"
            )

            if route.document_ids:

                st.write(
                    "**Selected documents:** "
                    + ", ".join(
                        route.document_ids
                    )
                )

            st.write(
                f"**Rewritten query:** "
                f"{response['rewritten_query']}"
            )

        # ---------------------------------
        # Retrieved sources
        # ---------------------------------

        with st.expander(
            "Retrieved evidence"
        ):

            for rank, result in enumerate(
                response[
                    "retrieved_results"
                ],
                start=1,
            ):

                chunk = result.chunk

                st.markdown(
                    f"""
**{rank}. {chunk.document_name}**

- Type: `{chunk.document_type}`
- Section: `{chunk.section}`
- Page: `{chunk.page}`
- Chunk ID: `{chunk.chunk_id}`
"""
                )

                st.write(
                    chunk.text
                )

                st.divider()