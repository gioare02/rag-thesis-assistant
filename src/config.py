"""
Central configuration for the RAG project.
"""

# Embedding model
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Chunking
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50

# Retrieval
TOP_K = 5

# LLM
LLM_MODEL_NAME = "gpt-5"