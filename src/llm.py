import os
from typing import List

from dotenv import load_dotenv
from openai import OpenAI

from src.models import Chunk
from src.config import LLM_MODEL_NAME


load_dotenv()


def create_client() -> OpenAI:
    """
    Create an OpenAI client using the API key stored in .env.
    """

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY not found. Add it to the .env file."
        )

    return OpenAI(api_key=api_key)


def build_context(
    chunks: List[Chunk],
) -> str:
    """
    Convert the final context chunks into structured context
    that can be passed to the LLM.

    These chunks may include both directly retrieved chunks
    and neighbouring chunks added during context expansion.
    """

    context_blocks: List[str] = []

    for source_number, chunk in enumerate(
        chunks,
        start=1,
    ):

        context_blocks.append(
            (
                f"[Source {source_number}]\n"
                f"Document: {chunk.document_name}\n"
                f"Document type: {chunk.document_type}\n"
                f"Page: {chunk.page}\n"
                f"Section: {chunk.section}\n\n"
                f"{chunk.text}"
            )
        )

    return "\n\n--------------------\n\n".join(
        context_blocks
    )


def generate_answer(
    question: str,
    context_chunks: List[Chunk],
    client: OpenAI,
    model: str = LLM_MODEL_NAME,
) -> str:
    """
    Generate an answer grounded only in the final context.
    """

    if not question.strip():
        raise ValueError(
            "question cannot be empty"
        )

    if not context_chunks:
        return (
            "I do not have enough information "
            "in the provided documents."
        )

    context = build_context(
        context_chunks
    )

    prompt = f"""
You are an academic research assistant.

Answer the user's question using only the retrieved context below.

Rules:
- Do not use external knowledge.
- If the context is insufficient, clearly say so.
- Cite relevant evidence using [Source 1], [Source 2], etc.
- Do not mention retrieval or reranker scores.
- Distinguish clearly between evidence from the thesis and evidence from external papers when relevant.
- Keep the answer clear, precise, and concise.

Retrieved context:
{context}

Question:
{question}
"""

    response = client.responses.create(
        model=model,
        input=prompt,
    )

    return response.output_text.strip()


def rewrite_query(
    question: str,
    chat_history: List[dict],
    client: OpenAI,
    model: str = LLM_MODEL_NAME,
) -> str:
    """
    Rewrite a conversational follow-up question into
    a standalone retrieval query.

    Example:

        Previous question:
        "What did the thesis find about volatility?"

        Follow-up:
        "Was this effect stable?"

    becomes approximately:

        "Was the thesis result on sentiment and
        volatility stable over time?"
    """

    if not question.strip():
        raise ValueError(
            "question cannot be empty"
        )

    # If there is no conversation history,
    # the question is already standalone.
    if not chat_history:
        return question.strip()

    history_lines = []

    # Only use the most recent messages.
    for message in chat_history[-6:]:

        role = message.get(
            "role",
            "unknown",
        )

        content = message.get(
            "content",
            "",
        )

        history_lines.append(
            f"{role}: {content}"
        )

    history_text = "\n".join(
        history_lines
    )

    prompt = f"""
You are the query-rewriting component of an academic retrieval system.

Rewrite the user's latest question as a concise standalone search query.

Rules:
- Preserve the user's original meaning.
- Resolve references such as "it", "this", "that result",
  or "the previous method" using the conversation when possible.
- Do not answer the question.
- Do not add information that is not present in the conversation.
- If the latest question is already standalone,
  return it unchanged or only minimally cleaned up.

Conversation:
{history_text}

Latest question:
{question}

Standalone retrieval query:
"""

    response = client.responses.create(
        model=model,
        input=prompt,
    )

    rewritten_query = (
        response.output_text.strip()
    )

    if not rewritten_query:
        return question.strip()

    return rewritten_query


if __name__ == "__main__":

    from src.embeddings import (
        load_embedding_model,
    )

    from src.vector_store import (
        load_vector_store,
    )

    from src.hybrid_retriever import (
        hybrid_retrieve,
    )

    from src.reranker import (
        load_reranker,
        rerank_results,
    )

    from src.context_expansion import (
        expand_context,
    )

    # -------------------------
    # Load resources
    # -------------------------

    client = create_client()

    embedding_model = (
        load_embedding_model()
    )

    reranker = load_reranker()

    index, chunks = (
        load_vector_store()
    )

    # -------------------------
    # Example question
    # -------------------------

    question = (
        "Does sentiment improve short-term "
        "volatility forecasting?"
    )

    # -------------------------
    # Hybrid retrieval
    # -------------------------

    hybrid_results = hybrid_retrieve(
        query=question,
        model=embedding_model,
        index=index,
        chunks=chunks,
        top_k=15,
        document_type="thesis",
    )

    # -------------------------
    # CrossEncoder reranking
    # -------------------------

    reranked_results = rerank_results(
        query=question,
        results=hybrid_results,
        reranker=reranker,
        top_k=5,
    )

    # -------------------------
    # Context expansion
    # -------------------------

    context_chunks = expand_context(
        results=reranked_results,
        chunks=chunks,
    )

    # -------------------------
    # Inspect retrieved results
    # -------------------------

    print("Reranked results")
    print("=" * 80)

    for rank, result in enumerate(
        reranked_results,
        start=1,
    ):

        chunk = result.chunk

        print(f"Result {rank}")

        print(
            f"Document: "
            f"{chunk.document_name}"
        )

        print(
            f"Type: "
            f"{chunk.document_type}"
        )

        print(
            f"Page: "
            f"{chunk.page}"
        )

        print(
            f"Section: "
            f"{chunk.section}"
        )

        print(
            f"Reranker score: "
            f"{result.score:.4f}"
        )

        print(
            chunk.text[:500]
        )

        print("-" * 80)

    # -------------------------
    # Inspect expanded context
    # -------------------------

    print()
    print(
        f"Expanded context: "
        f"{len(context_chunks)} chunks "
        f"from {len(reranked_results)} "
        f"reranked results"
    )

    print()

    for chunk in context_chunks:

        print(
            f"Chunk {chunk.chunk_id} | "
            f"Page {chunk.page} | "
            f"Section: {chunk.section}"
        )

    # -------------------------
    # Generate final answer
    # -------------------------

    answer = generate_answer(
        question=question,
        context_chunks=context_chunks,
        client=client,
    )

    print()
    print("Answer")
    print("=" * 80)
    print(answer)