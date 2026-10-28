"""RAG 编排层：检索 -> 拼 prompt -> 流式生成 -> 记录日志。

把「流程」和「组件」分开：
组件（切分、检索、LLM）各自可单测；这里只负责串起来，方便替换任意一环。
"""

from __future__ import annotations

import time
from typing import AsyncIterator

from .config import Settings
from .db import Repository
from .ingest.chunker import chunk_text, index_text
from .ingest.loaders import extract_text
from .llm import SYSTEM_PROMPT, LLMClient, LLMError
from .models import Source
from .rerank import build_reranker
from .retrieval import HybridRetriever

MAX_CONTEXT_CHARS = 3000

_RULES = (
    "请遵守以下规则：\n"
    "1. 只依据【资料】回答；资料中没有的信息，直接回答「资料中未提及」，不要编造。\n"
    "2. 每个结论后面标注来源编号，例如 [1]、[2]；编号必须是资料里出现过的。\n"
    "3. 先给结论，再给依据，总长度控制在 300 字以内。"
)

_REWRITE_SYSTEM = (
    "你是查询改写助手。用户会给你一段对话历史和一个追问，"
    "请把追问改写成可以独立检索的完整问题：补全指代词（它、这个、那呢），"
    "保留关键实体，不要回答问题，不要解释。只输出改写后的问题。"
)


class RagService:
    def __init__(self, settings: Settings, repo: Repository, llm: LLMClient) -> None:
        self.settings = settings
        self.repo = repo
        self.llm = llm
        self.retriever = HybridRetriever(
            rrf_k=settings.rrf_k, embed_provider=settings.embedding_provider
        )
        self.reranker = build_reranker(settings.rerank_provider, llm)
        self._chunks: dict[str, dict] = {}

    # ---------- 索引 ----------
    def refresh_index(self) -> None:
        rows = self.repo.iter_chunks()
        self._chunks = {row["id"]: row for row in rows}
        items = [(row["id"], index_text(row["section"], row["text"])) for row in rows]
        self.retriever.build(items)

    # ---------- 入库 ----------
    def ingest(self, filename: str, data: bytes) -> tuple[dict, str, bool]:
        """解析并入库。同名文档会被替换（返回 replaced=True），避免重复上传产生多份。"""
        text = extract_text(filename, data)
        pieces = chunk_text(text, self.settings.chunk_size, self.settings.chunk_overlap)
        if not pieces:
            raise ValueError("切分结果为空，请检查文档内容")

        replaced = False
        existing = self.repo.find_by_name(filename)
        if existing:
            self.repo.delete_document(existing.id)
            replaced = True

        doc = self.repo.add_document(
            name=filename,
            ext=filename.rsplit(".", 1)[-1].lower() if "." in filename else "",
            size=len(data),
            chunks=[(p.ordinal, p.section, p.text) for p in pieces],
        )
        self.refresh_index()
        return doc.model_dump(), pieces[0].text[:200], replaced

    # ---------- 检索 ----------
    def retrieve(self, question: str, top_k: int | None = None) -> list[Source]:
        """粗排（BM25 + 向量 + RRF 融合），同步、无外部调用。"""
        k = top_k or self.settings.top_k
        hits = self.retriever.search(
            question,
            top_k=self.settings.rerank_candidates or k,
            bm25_weight=self.settings.bm25_weight,
            vector_weight=self.settings.vector_weight,
            bm25_min_ratio=self.settings.bm25_min_ratio,
            min_cos_ratio=self.settings.min_cos_ratio,
        )
        sources: list[Source] = []
        for rank, (chunk_id, score) in enumerate(hits, start=1):
            row = self._chunks.get(chunk_id)
            if not row:
                continue
            sources.append(
                Source(
                    rank=rank,
                    chunk_id=chunk_id,
                    doc_name=row["doc_name"],
                    section=row["section"],
                    text=row["text"],
                    score=round(score, 4),
                )
            )
        return sources

    async def aretrieve(self, question: str, top_k: int | None = None) -> list[Source]:
        """粗排 + 可选精排。重排是网络调用，所以走异步。"""
        k = top_k or self.settings.top_k
        sources = self.retrieve(question, top_k=None)
        return await self.reranker.rerank(question, sources, k)

    # ---------- 生成 ----------
    def build_messages(
        self, question: str, sources: list[Source], history: list | None = None
    ) -> list[dict]:
        blocks = [
            f"[{s.rank}] 《{s.doc_name}》· {s.section}\n{s.text}" for s in sources
        ]
        context = "\n\n".join(blocks)[:MAX_CONTEXT_CHARS]
        user = f"【资料】\n{context}\n\n【问题】\n{question}\n\n{_RULES}"

        messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
        # 最近 4 轮对话作为上下文，帮助模型理解"它/这个"之类的指代
        for turn in (history or [])[-4:]:
            messages.append({"role": turn.role, "content": turn.content[:300]})
        messages.append({"role": "user", "content": user})
        return messages

    async def rewrite(self, question: str, history: list) -> str:
        """把追问改写成可独立检索的问题。

        没有历史时直接返回原问题（不浪费一次调用）。
        改写失败一律退回启发式拼接，保证多轮能力不会因为模型抖动而整体不可用。
        """
        if not history:
            return question

        last_user = next(
            (t.content for t in reversed(history) if t.role == "user"), ""
        )
        fallback = f"{last_user} {question}".strip() if last_user else question

        try:
            text = await self.llm.complete(
                [
                    {"role": "system", "content": _REWRITE_SYSTEM},
                    {
                        "role": "user",
                        "content": f"追问：{question}\n上一轮问题：{last_user}",
                    },
                ]
            )
            text = (text or "").strip().strip('"“”').split("\n")[0].strip()
            if 2 <= len(text) <= 120:
                return text
        except LLMError:
            pass
        return fallback

    async def astream(
        self,
        question: str,
        top_k: int | None = None,
        history: list | None = None,
    ) -> AsyncIterator[dict]:
        """产出事件流：rewrite? -> sources -> delta* -> done（或 error）。"""
        started = time.perf_counter()
        history = history or []

        search_query = question
        if history:
            search_query = await self.rewrite(question, history)
            if search_query != question:
                yield {"type": "rewrite", "question": search_query}

        sources = await self.aretrieve(search_query, top_k)
        yield {"type": "sources", "sources": [s.model_dump() for s in sources]}

        if not sources:
            answer = "知识库中还没有可用的文档，请先上传资料后再提问。"
            for i in range(0, len(answer), 12):
                yield {"type": "delta", "text": answer[i : i + 12]}
            yield {"type": "done", "latency_ms": 0, "provider": self.llm.name, "n_sources": 0}
            return

        messages = self.build_messages(search_query, sources, history)
        parts: list[str] = []
        try:
            async for piece in self.llm.stream(messages, refs=[s.text for s in sources]):
                parts.append(piece)
                yield {"type": "delta", "text": piece}
        except LLMError as exc:
            yield {"type": "error", "message": str(exc)}
            return

        latency_ms = int((time.perf_counter() - started) * 1000)
        answer = "".join(parts)
        self.repo.log_qa(question, answer, len(sources), latency_ms, self.llm.name)
        yield {
            "type": "done",
            "latency_ms": latency_ms,
            "provider": self.llm.name,
            "n_sources": len(sources),
        }
