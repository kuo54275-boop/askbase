"""重排层单测：打分解析、重排序、失败降级。"""

from __future__ import annotations

import asyncio

from app.llm import LLMError, MockClient
from app.models import Source
from app.rerank import LLMReranker, NoopReranker, _parse_scores, build_reranker


def _src(rank: int, doc: str) -> Source:
    return Source(
        rank=rank, chunk_id=f"c{rank}", doc_name=doc, section="s", text=f"内容{rank}", score=0.1
    )


def test_parse_scores_handles_noise():
    raw = "好的，结果如下：\n1: 9\n[2]：3\n3 - 7\n无效行\n4: 11"
    scores = _parse_scores(raw)
    assert scores[1] == 9
    assert scores[2] == 3
    assert 3 not in scores  # "3 - 7" 不是合法格式
    assert 4 not in scores  # 超出 0-10


class _FakeLLM:
    name = "deepseek"

    def __init__(self, reply: str) -> None:
        self.reply = reply

    async def complete(self, messages, max_tokens: int = 80) -> str:  # noqa: ARG002
        return self.reply

    async def stream(self, messages, refs=None):  # pragma: no cover - 协议占位
        yield ""


def test_llm_reranker_reorders_by_score():
    sources = [_src(1, "A"), _src(2, "B")]
    reranker = LLMReranker(_FakeLLM("1: 2\n2: 9"))
    result = asyncio.run(reranker.rerank("问题", sources, top_k=2))
    assert [s.doc_name for s in result] == ["B", "A"]
    assert [s.rank for s in result] == [1, 2]


def test_llm_reranker_falls_back_on_error():
    class Boom(_FakeLLM):
        async def complete(self, messages, max_tokens: int = 80) -> str:
            raise LLMError("timeout")

    sources = [_src(1, "A"), _src(2, "B")]
    reranker = LLMReranker(Boom(""))
    result = asyncio.run(reranker.rerank("问题", sources, top_k=2))
    assert [s.doc_name for s in result] == ["A", "B"]


def test_llm_reranker_falls_back_on_garbage():
    sources = [_src(1, "A"), _src(2, "B")]
    reranker = LLMReranker(_FakeLLM("我不知道"))
    result = asyncio.run(reranker.rerank("问题", sources, top_k=2))
    assert [s.doc_name for s in result] == ["A", "B"]


def test_noop_reranker_keeps_order():
    sources = [_src(1, "A"), _src(2, "B")]
    result = asyncio.run(NoopReranker().rerank("问题", sources, top_k=1))
    assert len(result) == 1
    assert result[0].doc_name == "A"


def test_build_reranker_degrades_for_mock():
    # Mock 模式下没必要求助模型，直接不重排
    assert build_reranker("llm", MockClient()).name == "none"
    assert build_reranker("none", MockClient()).name == "none"
