from typing import List, Optional

import faiss
from openai import OpenAI
from sentence_transformers import (
    SentenceTransformer,
    CrossEncoder,
)

from src.models import Chunk
from src.llm import (
    rewrite_query,
    generate_answer,
)
from src.query_router import (
    QueryRoute,
    route_query,
)
from src.hybrid_retriever import (
    hybrid_retrieve,
)
from src.reranker import (
    rerank_results,
)
from src.context_expansion import (
    expand_context,
)


class RAGPipeline:
    """
    End-to-end orchestration of the RAG system.

    Pipeline:

        conversational question
            ↓
        query rewriting
            ↓
        query routing
            ↓
        hybrid retrieval
            ↓
        CrossEncoder reranking
            ↓
        context expansion
            ↓
        grounded generation
    """

    def __init__(
        self,
        client: OpenAI,
        embedding_model: SentenceTransformer,
        reranker: CrossEncoder,
        index: faiss.Index,
        chunks: List[Chunk],
    ):
        self.client = client

        self.embedding_model = (
            embedding_model
        )

        self.reranker = reranker

        self.index = index

        self.chunks = chunks


    def retrieve(
        self,
        query: str,
        route: QueryRoute,
        candidate_k: int = 15,
        final_k: int = 5,
    ):
        """
        Retrieve and rerank chunks according
        to the route selected by the router.
        """

        hybrid_results = hybrid_retrieve(
            query=query,
            model=self.embedding_model,
            index=self.index,
            chunks=self.chunks,
            top_k=candidate_k,
            document_type=route.document_type,
            document_ids=(
                route.document_ids
                or None
            ),
        )

        reranked_results = (
            rerank_results(
                query=query,
                results=hybrid_results,
                reranker=self.reranker,
                top_k=final_k,
            )
        )

        return reranked_results


    def run(
        self,
        question: str,
        chat_history: Optional[
            List[dict]
        ] = None,
    ) -> dict:
        """
        Run the complete RAG pipeline for
        one user question.

        Returns structured information that
        can later be displayed by Streamlit.
        """

        if not question.strip():
            raise ValueError(
                "question cannot be empty"
            )

        if chat_history is None:
            chat_history = []

        # ----------------------------------
        # 1. Query rewriting
        # ----------------------------------

        rewritten_query = rewrite_query(
            question=question,
            chat_history=chat_history,
            client=self.client,
        )

        # ----------------------------------
        # 2. Query routing
        # ----------------------------------

        route = route_query(
            question=rewritten_query,
            chunks=self.chunks,
            client=self.client,
        )

        # ----------------------------------
        # 3. Retrieval + reranking
        # ----------------------------------

        reranked_results = (
            self.retrieve(
                query=rewritten_query,
                route=route,
            )
        )

        # ----------------------------------
        # 4. Context expansion
        # ----------------------------------

        context_chunks = expand_context(
            results=reranked_results,
            chunks=self.chunks,
        )

        # ----------------------------------
        # 5. Grounded generation
        # ----------------------------------

        answer = generate_answer(
            question=question,
            context_chunks=context_chunks,
            client=self.client,
        )

        # ----------------------------------
        # 6. Structured response
        # ----------------------------------

        return {
            "answer": answer,

            "original_query": question,

            "rewritten_query": (
                rewritten_query
            ),

            "route": route,

            "retrieved_results": (
                reranked_results
            ),

            "context_chunks": (
                context_chunks
            ),
        }