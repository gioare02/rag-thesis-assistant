import streamlit as st
from src.config import TOP_K
from src.embeddings import load_embedding_model
from src.llm import create_client, generate_answer
from src.retriever import retrieve_chunks
from src.vector_store import load_vector_store

st.set_page_config(
    page_title="ThesisRAG",
    page_icon="📚",
    layout="wide",
)

@st.cache_resource
def load_resources():
    """
    Load expensive resources only once per Streamlit session.
    """
    embedding_model = load_embedding_model()
    index, chunks = load_vector_store()
    client = create_client()

    return embedding_model, index, chunks, client

st.title("📚 ThesisRAG")
with st.sidebar:
    st.header("Settings")

    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()
st.caption(
    "Ask questions about the thesis. "
    "Answers are generated only from retrieved document context."
)

try:
    embedding_model, index, chunks, client = load_resources()
except Exception as error:
    st.error(f"Could not load the RAG system: {error}")
    st.stop()


if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant" and "sources" in message:
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
                        f"Similarity score: {source['score']:.4f}"
                    )

                    st.write(source["text"])
                    st.divider()

question = st.chat_input("Ask a question about the thesis")

if question:
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching the thesis..."):
            results = retrieve_chunks(
                query=question,
                model=embedding_model,
                index=index,
                chunks=chunks,
                top_k=TOP_K,
            )

            answer = generate_answer(
                question=question,
                results=results,
                client=client,
            )

        st.markdown(answer)

        with st.expander("Retrieved sources"):
            for rank, result in enumerate(results, start=1):
                st.markdown(
                    f"**Source {rank} — "
                    f"{result.chunk.document}, "
                    f"page {result.chunk.page}**"
                )

                st.caption(
                    f"Similarity score: {result.score:.4f}"
                )

                st.write(result.chunk.text)

                st.divider()

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
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