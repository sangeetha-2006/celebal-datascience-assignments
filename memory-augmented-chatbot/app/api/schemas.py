"""
Step 7a: API request/response schemas.
"""
from pydantic import BaseModel


class ChatRequest(BaseModel):
    user_id: str
    query: str


class ChatResponse(BaseModel):
    answer: str
    used_rag: bool
    used_kg: bool
    used_tool: bool


class MemoryResponse(BaseModel):
    preferences: dict
    recent_history: list[dict]


class RagResult(BaseModel):
    chunk_id: str
    source_url: str
    title: str | None = None
    text: str
    score: float


class KgResult(BaseModel):
    source: str
    relation: str
    target: str
    source_url: str | None = None
