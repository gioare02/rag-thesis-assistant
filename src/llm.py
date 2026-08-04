import os
from typing import List

from dotenv import load_dotenv
from openai import OpenAI

from models import SearchResult
from config import LLM_MODEL_NAME


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


def build_context(results: List[SearchResult]) -> str:
    """
    Convert retrieved search results into structured context.
    """

    context_blocks: List[str] = []

    for rank, result in enumerate(results, start=1):
        chunk = result.chunk

        context_blocks.append(
            (
                f"[Source {rank}]\n"
                f"Document: {chunk.document}\n"
                f"Page: {chunk.page}\n"
                f"Similarity score: {result.score:.4f}\n\n"
                f"{chunk.text}"
            )
        )

    return "\n\n--------------------\n\n".join(context_blocks)


def generate_answer(
    question: str,
    results: List[SearchResult],
    client: OpenAI,
    model: str = LLM_MODEL_NAME,
) -> str:
    """
    Generate an answer grounded in the retrieved context.
    """

    if not question.strip():
        raise ValueError("question cannot be empty")

    if not results:
        return "I do not have enough information in the provided documents."

    context = build_context(results)

    prompt = f"""
        You are an academic research assistant.

        Answer the question using only the context provided below.

        Rules:
        - Do not use external knowledge.
        - If the context is insufficient, clearly say so.
        - Cite the relevant source using the format [Source 1], [Source 2], etc.
        - Do not mention similarity scores in the answer.
        - Keep the answer clear and concise.

        Context:
        {context}

        Question:
        {question}
        """

    response = client.responses.create(
        model=model,
        input=prompt,
    )

    return response.output_text

if __name__ == "__main__":
    from embeddings import load_embedding_model
    from retriever import retrieve_chunks
    from vector_store import load_vector_store

    client = create_client()
    embedding_model = load_embedding_model()
    index, chunks = load_vector_store()

    question = "Why was FinBERT used for sentiment analysis?"

    results = retrieve_chunks(
        query=question,
        model=embedding_model,
        index=index,
        chunks=chunks,
        top_k=5,
    )

    print("Retrieved sources")
    print("=" * 80)

    for result in results:
        print(
            f"Page {result.chunk.page} | "
            f"Score {result.score:.4f}"
        )
        print(result.chunk.text[:300])
        print("-" * 80)

    answer = generate_answer(
        question=question,
        results=results,
        client=client,
    )

    print()
    print("Answer")
    print("=" * 80)
    print(answer)

'''
User
   │
   ▼
"Why FinBERT?"
 │
 ▼
Embedding della domanda
   │
   ▼
Retriever (FAISS) → trova i documenti più pertinenti.
   │
   ▼
5 RELEVANT chunk
   │
   ▼
build_context()
   │
   ▼
Prompt
   │
   ▼
LLM (GPT) → legge quei documenti e scrive una risposta naturale.
   │
   ▼
Response

'''