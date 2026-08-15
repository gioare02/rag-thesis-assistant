from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Page:
    document_id: str
    document_name: str
    document_type: str
    page: int
    text: str
    section: Optional[str] = None


@dataclass
class Chunk:
    chunk_id: int

    document_id: str
    document_name: str
    document_type: str

    page: int
    text: str

    section: Optional[str] = None

    previous_chunk_id: Optional[int] = None
    next_chunk_id: Optional[int] = None


@dataclass
class SearchResult:
    chunk: Chunk
    score: float


@dataclass

class QueryRoute:
    intent: str
    document_type: Optional[str] = None
    document_ids: Optional[List[str]] = None
    reasoning: Optional[str] = None