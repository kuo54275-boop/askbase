"""重排（rerank）：对粗排结果再做一次精排。

为什么需要：检索是"召回优先"的，top-5 里通常有正确切片但排不在第一。
消融里 Recall@3 = 100% 而 Recall@1 只有 94.34% —— 那 5.66% 就是重排要吃掉的空间。

提供两种实现：
- NoopReranker：不重排，用于对照实验；
- LLMReranker：让模型给候选打分排序（listwise）。不需要下载模型、不占磁盘，
  代价是每个 query 多一次 API 调用（约 0.5~1s）。
  后续可以平替成 bge-reranker 这类 cross-encoder，接口不变。
"""

from __future__ import annotations

import re
from typing import Protocol

from .llm import LLMClient, LLMError
from .models import Source

_SYSTEM = (
    "你是一个检索结果重排助手。给定用户问题和若干候选片段，"
    "请判断每个片段与问题的相关性，打分 0-10（10 = 直接回答该问题，0 = 完全无关）。\n"
    "只输出每行一个「编号:分数」，不要输出任何解释。例如：\n1: 9\n2: 3"
)

_MAX_SNIPPET = 220


class Reranker(Protocol):
    name: str

    async def rerank(self, question: str, sources: list[Source], top_k: int) -> list[Source]: ...


class NoopReranker:
    """不重排，保持检索顺序。"""

    name = "none"

    async def rerank(self, question: str, sources: list[Source], top_k: int) -> list[Source]:
        return sources[:top_k]


class LLMReranker:
    name = "llm"

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    async def rerank(self, question: str, sources: list[Source], top_k: int) -> list[Source]:
        if len(sources) <= 1:
            return sources[:top_k]

        listing = "\n\n".join(
            f"[{s.rank}] 《{s.doc_name}》· {s.section}\n{s.text[:_MAX_SNIPPET]}"
            for s in sources
        )
        messages = [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": f"问题：{question}\n\n候选片段：\n{listing}"},
        ]

        try:
            raw = await self.llm.complete(messages)
        except LLMError:
            # 重排失败不能影响主流程：退回原顺序
            return sources[:top_k]

        scores = _parse_scores(raw)
        if not scores:
            return sources[:top_k]

        # 未打分的候选排在已打分之后，保持原有相对顺序（稳定排序）
        ordered = sorted(
            sources,
            key=lambda s: (-scores.get(s.rank, -1), s.rank),
        )
        result = ordered[:top_k]
        for i, src in enumerate(result, start=1):
            # 用模型打分覆盖融合分：否则重排后会出现"第 1 名相关度低于第 2 名"
            src.score = round(scores.get(src.rank, 0.0), 1)
            src.rank = i
        return result


def _parse_scores(raw: str) -> dict[int, float]:
    """解析「编号:分数」，容忍模型输出多余文字或 markdown。"""
    scores: dict[int, float] = {}
    for line in raw.splitlines():
        match = re.match(r"\s*\[?(\d{1,2})\]?\s*[:：]\s*(\d{1,2})", line)
        if not match:
            continue
        idx, score = int(match.group(1)), float(match.group(2))
        if 0 <= score <= 10:
            scores[idx] = score
    return scores


def build_reranker(provider: str, llm: LLMClient) -> Reranker:
    if provider in ("llm", "deepseek") and getattr(llm, "name", "") != "mock":
        return LLMReranker(llm)
    return NoopReranker()
