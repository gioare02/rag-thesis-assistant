# src/query_router.py

import json
from typing import List, Optional

from openai import OpenAI

from src.models import Chunk, QueryRoute
from src.config import LLM_MODEL_NAME


def build_document_catalog(
    chunks: List[Chunk],
) -> str:
    """
    Build a compact description of the documents
    currently available in the knowledge base.

    We include each document only once.
    Useful for router to know which docuemnt really exists.
    """

    seen_document_ids = set()
    catalog_lines = []

    for chunk in chunks:

        if chunk.document_id in seen_document_ids:
            continue

        seen_document_ids.add(
            chunk.document_id
        )

        catalog_lines.append(
            (
                f"- id={chunk.document_id} | "
                f"name={chunk.document_name} | "
                f"type={chunk.document_type}"
            )
        )

    return "\n".join(
        catalog_lines
    )


def route_query(
    question: str,
    chunks: List[Chunk],
    client: OpenAI,
    model: str = LLM_MODEL_NAME,
) -> QueryRoute:
    """
    Classify the user's query and determine
    the appropriate retrieval scope.

    Supported intents:

    FACTUAL
        Ask for information, explanation,
        methodology, results, definitions, etc.

    COMPARE
        Compare the thesis with one or more papers,
        or compare multiple documents.

    VALIDATE
        Check whether available evidence supports
        or contradicts a claim.
    """

    if not question.strip():
        raise ValueError(
            "question cannot be empty"
        )

    document_catalog = (
        build_document_catalog(chunks)
    )

    prompt = f"""
You are the query-routing component of an academic RAG system.

Your job is NOT to answer the question.

Your job is to decide:

1. What type of query this is.
2. Which documents should be searched.

Possible intents:

FACTUAL
- The user asks for a fact, result, explanation,
  methodology, definition, or summary.

COMPARE
- The user asks to compare, contrast, or identify
  similarities or differences between sources.

VALIDATE
- The user asks whether a claim is supported,
  contradicted, confirmed, or challenged by the
  available evidence.

Retrieval scope rules:

- document_type = "thesis"
  when the question is clearly only about the thesis.

- document_type = "paper"
  when the question is clearly only about external papers.

- document_type = null
  when both thesis and papers may be relevant.

- document_ids should contain specific document IDs
  only when the user explicitly refers to identifiable
  documents from the catalog.

- Never invent document IDs.

Available documents:

{document_catalog}

User question:

{question}

Return ONLY valid JSON with exactly this structure:

{{
    "intent": "FACTUAL",
    "document_type": "thesis",
    "document_ids": [],
    "reasoning": "Short explanation."
}}
"""

    response = client.responses.create(
        model=model,
        input=prompt,
    )

    raw_output = (
        response.output_text.strip()
    )

    try:

        data = json.loads(
            raw_output
        )

    except json.JSONDecodeError as exc:

        raise ValueError(
            "Query router returned invalid JSON: "
            f"{raw_output}"
        ) from exc

    intent = str(
        data.get("intent", "")
    ).upper()

    valid_intents = {
        "FACTUAL",
        "COMPARE",
        "VALIDATE",
    }

    if intent not in valid_intents:
        raise ValueError(
            f"Invalid router intent: {intent}"
        )

    document_type = data.get(
        "document_type"
    )

    valid_document_types = {
        None,
        "thesis",
        "paper",
    }

    if document_type not in valid_document_types:
        document_type = None

    available_document_ids = {
        chunk.document_id
        for chunk in chunks
    }

    requested_document_ids = (
        data.get("document_ids")
        or []
    )

    # Important:
    # never trust document IDs invented by the LLM.
    document_ids = [
        document_id
        for document_id
        in requested_document_ids
        if document_id
        in available_document_ids
    ]

    reasoning = data.get(
        "reasoning"
    )

    return QueryRoute(
        intent=intent,
        document_type=document_type,
        document_ids=document_ids,
        reasoning=reasoning,
    )


if __name__ == "__main__":

    from src.llm import (
        create_client,
    )

    from src.vector_store import (
        load_vector_store,
    )

    client = create_client()

    _, chunks = (
        load_vector_store()
    )

    test_questions = [
        (
            "What did my thesis conclude "
            "about short-term volatility?"
        ),
        (
            "Compare my thesis with the "
            "uploaded papers on sentiment."
        ),
        (
            "Do the papers support my claim "
            "that sentiment is mainly useful "
            "for short-term volatility?"
        ),
    ]

    for question in test_questions:

        route = route_query(
            question=question,
            chunks=chunks,
            client=client,
        )

        print("=" * 80)

        print(
            f"Question: {question}"
        )

        print(
            f"Intent: "
            f"{route.intent}"
        )

        print(
            f"Document type: "
            f"{route.document_type}"
        )

        print(
            f"Document IDs: "
            f"{route.document_ids}"
        )

        print(
            f"Reasoning: "
            f"{route.reasoning}"
        )