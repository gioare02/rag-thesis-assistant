import os
from typing import List

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


def create_client() -> OpenAI:
    """
    Create an OpenAI client using the API key stored in .env.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key is None:
        raise ValueError(
            "OPENAI_API_KEY not found in .env"
        )
    return OpenAI(api_key=api_key)


def build_context(chunks: List[dict]) -> str:
    """
    Convert retrieved chunks into a prompt context.
    """
    context = []
    for chunk in chunks:
        context.append(
            f"""
Document: {chunk['document']}
Page: {chunk['page']}

{chunk['text']}
"""
        )
    # concatena tutti i chunk in un’unica stringa.
    return "\n\n--------------------\n\n".join(context)


def generate_answer(
    question: str,
    chunks: List[dict],
    client: OpenAI,
    model: str = "gpt-5",
) -> str:
    """
    Generate an answer grounded on the retrieved chunks.
    """

    context = build_context(chunks)
    prompt = f"""
You are an academic assistant.

Answer ONLY using the provided context.

If the answer is not contained in the context,
say that you do not have enough information.

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
    question = "What is the main contribution of this thesis?"

    retrieved_chunks = retrieve_chunks(
        query=question,
        model=embedding_model,
        index=index,
        chunks=chunks,
        top_k=5,
    )

    answer = generate_answer(
        question=question,
        chunks=retrieved_chunks,
        client=client,
    )

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