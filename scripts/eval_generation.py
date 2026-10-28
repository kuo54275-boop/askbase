"""生成质量评测：忠实度、引用准确率、拒答率。

为什么要单独做这一层：
检索指标（Recall@k）只能证明"资料找对了"，证明不了"答案没编"。
RAG 最典型的事故是——检索全对，模型却把资料里没有的常识混进答案。

用法：
    python scripts/eval_generation.py                # 默认 20 条正样本 + 全部负样本
    python scripts/eval_generation.py --limit 10
    python scripts/eval_generation.py --positives-only

成本提示：每条正样本 = 1 次生成 + 1 次裁判；负样本 = 1 次生成。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import statistics
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from app.config import get_settings  # noqa: E402
from app.db import Repository  # noqa: E402
from app.evaluation import load_cases  # noqa: E402
from app.judge import LLMJudge, extract_citations, is_refusal  # noqa: E402
from app.llm import build_llm  # noqa: E402
from app.rag import RagService  # noqa: E402


async def run_case(service: RagService, question: str) -> tuple[str, list, float]:
    """走真实链路（含改写/重排/流式生成），返回答案、来源、耗时。"""
    started = time.perf_counter()
    answer_parts: list[str] = []
    sources: list = []
    async for event in service.astream(question):
        if event["type"] == "sources":
            sources = [s for s in event["sources"]]
        elif event["type"] == "delta":
            answer_parts.append(event["text"])
        elif event["type"] == "error":
            return f"[ERROR] {event['message']}", sources, 0.0
    return "".join(answer_parts), sources, (time.perf_counter() - started) * 1000


def to_source_objects(raw: list):
    from app.models import Source

    return [Source(**item) for item in raw]


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=20, help="正样本条数")
    parser.add_argument("--positives-only", action="store_true")
    parser.add_argument("--negatives-only", action="store_true", help="只跑拒答测试，省时间")
    args = parser.parse_args()

    settings = get_settings()
    repo = Repository(settings.db_path)
    repo.init_db()
    llm = build_llm(
        settings.effective_llm_provider,
        settings.deepseek_api_key,
        settings.deepseek_base_url,
        settings.chat_model,
        settings.llm_timeout,
        settings.llm_max_retries,
    )
    if llm.name == "mock":
        print("当前是 Mock 模式，生成质量评测无意义，请先配置 DEEPSEEK_API_KEY")
        return 1

    service = RagService(settings, repo, llm)
    service.refresh_index()
    judge = LLMJudge(llm)

    cases = [] if args.negatives_only else load_cases(BASE / "eval" / "qa_set.jsonl")[: args.limit]

    grounded_scores: list[int] = []
    relevant_scores: list[int] = []
    cite_total = 0
    cite_bad = 0
    latencies: list[float] = []
    problems: list[str] = []

    if cases:
        print(f"正样本：{len(cases)} 条（真实调用生成 + 裁判）…")
    for i, case in enumerate(cases, start=1):
        answer, raw_sources, latency = await run_case(service, case.question)
        latencies.append(latency)
        sources = to_source_objects(raw_sources)

        cites = extract_citations(answer)
        cite_total += len(cites)

        verdict = await judge.judge(case.question, sources, answer)
        if verdict is None:
            problems.append(f"[裁判失败] {case.question}")
            continue

        grounded_scores.append(verdict.grounded)
        relevant_scores.append(verdict.relevant)
        cite_bad += len([c for c in verdict.bad_citations if c in cites])

        if verdict.grounded < 2 or verdict.bad_citations:
            problems.append(
                f"[grounded={verdict.grounded} bad={verdict.bad_citations}] "
                f"{case.question} → {verdict.reason[:60]}"
            )
        print(f"  {i}/{len(cases)} grounded={verdict.grounded} relevant={verdict.relevant}")

    if cases:
        print("\n=== 正样本结果 ===")
        if grounded_scores:
            print(f"忠实度均分      : {statistics.fmean(grounded_scores):.2f} / 2.00")
            print(
                f"完全忠实率       : {sum(1 for g in grounded_scores if g == 2) / len(grounded_scores):.1%}"
            )
            print(f"相关性均分      : {statistics.fmean(relevant_scores):.2f} / 2.00")
        if cite_total:
            print(f"引用准确率       : {1 - cite_bad / cite_total:.1%}（{cite_total} 个角标）")
        else:
            print("引用准确率       : 无法计算（答案里没有引用角标）")
        if latencies:
            print(f"平均端到端延迟    : {statistics.fmean(latencies):.0f} ms")

    if not args.positives_only:
        neg_path = BASE / "eval" / "negative_set.jsonl"
        negatives = [
            json.loads(line)["question"]
            for line in neg_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        print(f"\n负样本：{len(negatives)} 条（知识库里没有答案，应当拒答）…")
        refused = 0
        leaks: list[str] = []
        for q in negatives:
            answer, _src, _lat = await run_case(service, q)
            if is_refusal(answer):
                refused += 1
            else:
                leaks.append(f"{q} → {answer[:90]}")
        print(f"拒答率           : {refused / len(negatives):.1%}（{refused}/{len(negatives)}）")
        if leaks:
            print("未拒答（可能编造）：")
            for item in leaks:
                print(f"  - {item}")

    if problems:
        print("\n需要人工看一眼的样本：")
        for p in problems[:10]:
            print(f"  - {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
