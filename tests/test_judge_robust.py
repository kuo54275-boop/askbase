"""双序裁判单测：平均、反向角标映射、单侧失败降级。"""

from __future__ import annotations

import asyncio

from app.judge import JudgeResult, judge_robust, reorder_sources
from app.models import Source


def _src(rank: int, text: str) -> Source:
    return Source(
        rank=rank, chunk_id=f"c{rank}", doc_name=f"doc{rank}.md", section="s", text=text, score=1.0
    )


class _ScriptedJudge:
    """按调用顺序返回预设结果，模拟"正序给 2 分、倒序给 1 分"这类情况。"""

    def __init__(self, results: list[JudgeResult | None]) -> None:
        self.results = list(results)
        self.calls = 0

    async def judge(self, question, sources, answer):  # noqa: ARG002
        self.calls += 1
        if not self.results:
            return None
        return self.results.pop(0)


def _v(grounded: int, bad: list[int] | None = None) -> JudgeResult:
    return JudgeResult(grounded=grounded, relevant=2, bad_citations=bad or [], reason="r")


def test_reorder_sources_reverses_and_renumbers():
    sources = [_src(1, "a"), _src(2, "b"), _src(3, "c")]
    out = reorder_sources(sources, reverse=True)
    assert [s.text for s in out] == ["c", "b", "a"]
    assert [s.rank for s in out] == [1, 2, 3]
    # 原列表不能被改动
    assert [s.rank for s in sources] == [1, 2, 3]


def test_robust_averages_two_orders():
    judge = _ScriptedJudge([_v(2), _v(1)])
    verdict = asyncio.run(judge_robust(judge, "q", [_src(1, "a"), _src(2, "b")], "答案"))
    assert verdict is not None
    assert verdict.grounded == 1.5
    assert verdict.order_delta == 1
    assert judge.calls == 2


def test_reverse_bad_citation_maps_back():
    """倒序空间里的 [1] 对应原空间的最后一条。"""
    judge = _ScriptedJudge([_v(2, []), _v(2, [1])])
    sources = [_src(1, "a"), _src(2, "b"), _src(3, "c")]
    verdict = asyncio.run(judge_robust(judge, "q", sources, "答案"))
    assert verdict is not None
    assert verdict.bad_citations == [3]


def test_robust_falls_back_when_one_order_fails():
    judge = _ScriptedJudge([_v(2), None])
    verdict = asyncio.run(judge_robust(judge, "q", [_src(1, "a")], "答案"))
    assert verdict is not None
    assert verdict.grounded == 2.0
    assert verdict.order_delta == 0


def test_robust_returns_none_when_both_fail():
    judge = _ScriptedJudge([None, None])
    assert asyncio.run(judge_robust(judge, "q", [_src(1, "a")], "答案")) is None
