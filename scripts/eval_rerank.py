"""重排 A/B 评测：同一批问题，对比「不重排」与「LLM 重排」。

用法：
    python scripts/eval_rerank.py
    python scripts/eval_rerank.py --limit 20   # 只跑前 20 条，省时间

注意：会真实调用 53 次大模型（每条问题一次），产生少量费用。
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from app.config import get_settings  # noqa: E402
from app.db import Repository  # noqa: E402
from app.evaluation import load_cases  # noqa: E402
from app.llm import build_llm_from_settings  # noqa: E402
from app.rag import RagService  # noqa: E402

KS = (1, 3, 5)


async def run(service: RagService, cases, use_rerank: bool) -> dict:
    hits = {k: 0 for k in KS}
    rr: list[float] = []
    latencies: list[float] = []

    for case in cases:
        started = time.perf_counter()
        sources = (
            await service.aretrieve(case.question)
            if use_rerank
            else service.retrieve(case.question)
        )
        latencies.append((time.perf_counter() - started) * 1000)

        names = [s.doc_name for s in sources]
        rank = next((i for i, n in enumerate(names, start=1) if n == case.gold_doc), None)
        for k in KS:
            if rank and rank <= k:
                hits[k] += 1
        rr.append(1.0 / rank if rank else 0.0)

    total = len(cases) or 1
    return {
        "recall": {k: hits[k] / total for k in KS},
        "mrr": statistics.fmean(rr),
        "latency": statistics.fmean(latencies),
    }


def make_service(settings, repo, llm, rerank: bool) -> RagService:
    s = settings.model_copy(update={"rerank_provider": "llm" if rerank else "none"})
    service = RagService(s, repo, llm)
    service.refresh_index()
    return service


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 条")
    args = parser.parse_args()

    settings = get_settings()
    repo = Repository(settings.db_path)
    repo.init_db()
    llm = build_llm_from_settings(settings)
    if llm.name == "mock":
        print("当前是 Mock 模式，重排无意义，请先在 .env 配置 LLM_API_KEY")
        return 1

    cases = load_cases(BASE / "eval" / "qa_set.jsonl")
    if args.limit:
        cases = cases[: args.limit]

    base_service = make_service(settings, repo, llm, rerank=False)
    print(f"样本数：{len(cases)}，开始跑基线（不重排）…")
    baseline = await run(base_service, cases, use_rerank=False)

    rerank_service = make_service(settings, repo, llm, rerank=True)
    print("开始跑 LLM 重排…")
    reranked = await run(rerank_service, cases, use_rerank=True)

    def fmt(m: dict) -> str:
        return (
            f"Recall@1={m['recall'][1]:.2%}  Recall@3={m['recall'][3]:.2%}  "
            f"Recall@5={m['recall'][5]:.2%}  MRR={m['mrr']:.4f}  延迟={m['latency']:.0f}ms"
        )

    print("\n=== 结果 ===")
    print(f"不重排 : {fmt(baseline)}")
    print(f"LLM 重排: {fmt(reranked)}")
    delta = (reranked["recall"][1] - baseline["recall"][1]) * 100
    print(f"\nRecall@1 变化: {delta:+.2f} 个百分点")
    print(f"延迟增加   : {reranked['latency'] - baseline['latency']:+.0f} ms/次")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
