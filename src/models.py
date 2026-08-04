from dataclasses import dataclass


@dataclass
class Page:  # risultato di ingest.py
    document: str
    page: int
    text: str


@dataclass
class Chunk: # risultato del chunking.
    chunk_id: int
    document: str
    page: int
    text: str


@dataclass
class SearchResult: # quello che restituisce il retriever.
    chunk: Chunk
    score: float