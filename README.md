# 📚 ThesisRAG

A conversational Retrieval-Augmented Generation (RAG) application for querying academic documents using semantic search, lexical search and LLMs.

The application allows users to chat with a permanent knowledge base (e.g. a master's thesis) and dynamically uploaded research papers. It combines semantic retrieval (FAISS), lexical retrieval (BM25), Reciprocal Rank Fusion (RRF), reranking and query rewriting to generate grounded answers with source citations.

---

## Features

- 📄 PDF ingestion
- ✂️ Configurable text chunking with overlap
- 🔎 Semantic retrieval using Sentence Transformers + FAISS
- 📚 Lexical retrieval using LangChain BM25
- 🔀 Hybrid retrieval using Reciprocal Rank Fusion (RRF)
- 🎯 CrossEncoder reranking
- 💬 Conversational query rewriting
- 🤖 OpenAI GPT integration
- 📑 Source attribution with page references
- 📤 Upload additional PDF documents
- 🗂 Permanent and dynamic document collections
- 🖥 Interactive Streamlit interface

---

## Architecture

```

User Question
↓
Query Rewriting
↓
┌───────────────────────┐
│ Semantic Search (FAISS)
└───────────────────────┘
↓

┌───────────────────────┐
│ Lexical Search (BM25)
└───────────────────────┘
↓

Reciprocal Rank Fusion
↓

CrossEncoder Reranker
↓

GPT
↓

Grounded Answer + Sources