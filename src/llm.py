import os
from typing import List

from dotenv import load_dotenv
from openai import OpenAI

from src.models import Chunk
from src.config import LLM_MODEL_NAME
from src.query_router import route_query


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

def get_generation_instructions(
    intent: str,
) -> str:
    """
    Return generation instructions based on query intent.
    """

    if intent == "FACTUAL":
        return """
    Answer the question directly.

    - Give the answer first.
    - Explain the relevant evidence.
    - Cite the supporting sources.
    - Do not force a comparison.
    """

    if intent == "COMPARE":
        return """
    Explicitly compare the thesis with the external literature.

    Structure the answer around:
    1. What the thesis says.
    2. What the external paper(s) say.
    3. Main similarities.
    4. Main differences.
    5. Overall conclusion.

    Do not simply summarize each source independently.
    If evidence from one side is missing, clearly say so.
    """

    if intent == "VALIDATE":
        return """
    Evaluate whether the user's claim is supported by the available evidence.

    Structure the answer around:
    1. The claim being assessed.
    2. Supporting evidence.
    3. Contradicting or weakening evidence.
    4. Final assessment.

    Use exactly one of these final assessment labels:

    SUPPORTED
    PARTIALLY SUPPORTED
    MIXED EVIDENCE
    CONTRADICTED
    INSUFFICIENT EVIDENCE

    Do not treat absence of evidence as evidence of contradiction.
    """

    raise ValueError(
        f"Unsupported generation intent: {intent}"
    )


def generate_answer(
    question: str,
    context_chunks: List[Chunk],
    client: OpenAI,
    intent: str = "FACTUAL",
    model: str = LLM_MODEL_NAME,
) -> str:
    """
    Generate a grounded answer using an
    intent-specific generation strategy.
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

    intent_instructions = (
        get_generation_instructions(
            intent
        )
    )

    prompt = f"""
        You are an academic research assistant.

        Answer the user's question using ONLY the retrieved context below.

        General rules:
        - Do not use external knowledge.
        - Do not invent facts or evidence.
        - If the context is insufficient, clearly say so.
        - Cite relevant evidence using [Source 1], [Source 2], etc.
        - Do not mention retrieval, RRF, similarity, or reranker scores.
        - Clearly distinguish evidence from the thesis and evidence from external papers.
        - Prefer precise statements over broad claims.

        Query intent:
        {intent}

        Intent-specific instructions:
        {intent_instructions}

        Retrieved context:
        {context}

        User question:
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