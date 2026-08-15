from typing import List, Optional

import faiss
from openai import OpenAI
from sentence_transformers import (
    SentenceTransformer,
    CrossEncoder,
)

from src.models import (
    Chunk,
    QueryRoute,
)

from src.llm import (
    rewrite_query,
    generate_answer,
)

from src.query_router import (
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


    def retrieve_factual(
        self,
        query: str,
        route: QueryRoute,
        candidate_k: int = 15,
        final_k: int = 5,
    ):
        """
        Standard retrieval for factual questions.

        The router decides whether retrieval should search:
        - only the thesis
        - only papers
        - specific documents
        - the whole corpus
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

        reranked_results = rerank_results(
            query=query,
            results=hybrid_results,
            reranker=self.reranker,
            top_k=final_k,
        )

        return reranked_results


    def retrieve_from_source(
        self,
        query: str,
        document_type: str,
        candidate_k: int = 10,
    ):
        """
        Retrieve candidates from one document type only.

        Returns an empty list if that source type is not
        currently available in the knowledge base.
        """

        source_exists = any(
            chunk.document_type == document_type
            for chunk in self.chunks
        )

        if not source_exists:
            return []

        return hybrid_retrieve(
            query=query,
            model=self.embedding_model,
            index=self.index,
            chunks=self.chunks,
            top_k=candidate_k,
            document_type=document_type,
        )


    def retrieve_balanced(
        self,
        query: str,
        candidate_k_per_source: int = 10,
        final_k: int = 6,
    ):
        """
        Retrieve evidence separately from thesis and papers.

        This prevents global retrieval from returning mostly
        chunks from one source type.

        Pipeline:

            thesis retrieval
                +
            paper retrieval
                ↓
            merge candidates
                ↓
            CrossEncoder reranking
        """

        thesis_results = self.retrieve_from_source(
            query=query,
            document_type="thesis",
            candidate_k=candidate_k_per_source,
        )

        paper_results = self.retrieve_from_source(
            query=query,
            document_type="paper",
            candidate_k=candidate_k_per_source,
        )

        combined_results = (
            thesis_results
            + paper_results
        )

        if not combined_results:
            return []

        reranked_results = rerank_results(
            query=query,
            results=combined_results,
            reranker=self.reranker,
            top_k=final_k,
        )

        return reranked_results


    def retrieve(
        self,
        query: str,
        route: QueryRoute,
    ):
        """
        Select the retrieval strategy according to query intent.
        """

        if route.intent == "FACTUAL":

            return self.retrieve_factual(
                query=query,
                route=route,
            )

        if route.intent == "COMPARE":

            return self.retrieve_balanced(
                query=query,
                candidate_k_per_source=10,
                final_k=6,
            )

        if route.intent == "VALIDATE":

            return self.retrieve_balanced(
                query=query,
                candidate_k_per_source=10,
                final_k=6,
            )

        raise ValueError(
            f"Unsupported query intent: {route.intent}"
        )

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
            intent=route.intent,
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
        
if __name__ == "__main__":

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

    # -------------------------
    # Load resources
    # -------------------------

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

    # -------------------------
    # Build pipeline
    # -------------------------

    pipeline = RAGPipeline(
        client=client,
        embedding_model=embedding_model,
        reranker=reranker,
        index=index,
        chunks=chunks,
    )

    # -------------------------
    # Test question
    # -------------------------

    test_questions = [
        (
            "What did my thesis conclude "
            "about short-term volatility?"
        ),

        (
            "Compare the findings of my thesis "
            "with the uploaded papers on sentiment "
            "and volatility."
        ),

        (
            "Do the uploaded papers support "
            "my thesis claim that sentiment is "
            "mainly useful for short-term volatility?"
        ),
    ]


    for question in test_questions:

        print()
        print("#" * 100)
        print(f"QUESTION: {question}")
        print("#" * 100)
        print()

        response = pipeline.run(
            question=question
        )

        route = response["route"]

        print(
            f"Intent: {route.intent}"
        )

        print(
            f"Document type: "
            f"{route.document_type}"
        )

        print()

        print("Retrieved sources:")

        for rank, result in enumerate(
            response["retrieved_results"],
            start=1,
        ):

            chunk = result.chunk

            print(
                f"{rank}. "
                f"{chunk.document_type} | "
                f"{chunk.document_name} | "
                f"page {chunk.page} | "
                f"{chunk.section}"
            )

        print()

        print("ANSWER")
        print("-" * 80)

        print(
            response["answer"]
        )