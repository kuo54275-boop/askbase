"""请求 / 响应的数据结构（Pydantic 模型）。

统一出入口契约，写测试时直接复用。
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class DocMeta(BaseModel):
    id: str
    name: str
    ext: str
    size: int
    n_chunks: int
    created_at: str


class UploadResult(BaseModel):
    doc: DocMeta
    preview: str = Field(description="首个切片的前 200 字，便于前端确认解析正常")
    replaced: bool = Field(default=False, description="同名文档被覆盖时为 true")


class Chunk(BaseModel):
    id: str
    doc_id: str
    doc_name: str
    section: str
    ordinal: int
    text: str
    score: float = 0.0
    rank: int = 0


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=4000)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=20)
    history: list[ChatTurn] = Field(default_factory=list, max_length=20)


class Source(BaseModel):
    rank: int
    chunk_id: str
    doc_name: str
    section: str
    text: str
    score: float


class EvalCase(BaseModel):
    question: str
    gold_doc: str = Field(description="标准答案所在文档名，用于算 recall / MRR")
    answer_keywords: list[str] = Field(default_factory=list)


class EvalResult(BaseModel):
    total: int
    recall_at: dict[str, float]
    mrr: float
    avg_latency_ms: float
    details: list[dict]


class HealthInfo(BaseModel):
    status: Literal["ok"]
    llm_provider: str
    embedding_provider: str
    n_docs: int
    n_chunks: int
