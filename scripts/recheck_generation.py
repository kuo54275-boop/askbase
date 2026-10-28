"""生成评测复核：规则校验 + 跨模型裁判 + 位置偏差测试。

三重独立证据：

1. **规则校验**（app/verify.py）：完全不看模型，只检查「引用的资料是否支持那句话」。
2. **跨模型裁判**：同一批答案，分别用 deepseek-chat 和 deepseek-reasoner 打分。
   两个模型训练目标不同，如果结论一致，说明裁判本身是稳的。
3. **位置偏差**：把资料顺序倒过来再判一次。LLM 普遍偏爱靠前的材料，
   如果倒序后分数明显变化，说明之前的评分被位置影响了。

用法：
    python scripts/recheck_generation.py            # 10 条
    python scripts/recheck_generation.py --limit 20
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from app.config import get_settings  # noqa: E402
from app.db import Repository  # noqa: E402
from app.evaluation import load_cases  # noqa: E402
from app.judge import LLMJudge, judge_robust  # noqa: E402
from app.llm import build_judge_llm, build_llm, probe_client  # noqa: E402
from app.models import Source  # noqa: E402
from app.rag import RagService  # noqa: E402
from app.verify import verify_answer  # noqa: E402


def client(settings, model: str):
    return build_llm(
        settings.effective_llm_provider,
        settings.deepseek_api_key,
        settings.deepseek_base_url,
        model,
        settings.llm_timeout,
        settings.llm_max_retries,
    )


async def collect(service: RagService, question: str) -> tuple[str, list[Source]]:
    parts: list[str] = []
    raw: list[dict] = []
    async for ev in service.astream(question):
        if ev["type"] == "sources":
            raw = ev["sources"]
        elif ev["type"] == "delta":
            parts.append(ev["text"])
    return "".join(parts), [Source(**s) for s in raw]


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()

    settings = get_settings()
    repo = Repository(settings.db_path)
    repo.init_db()
    chat = client(settings, "deepseek-chat")
    reasoner = client(settings, "deepseek-reasoner")
    service = RagService(settings, repo, chat)
    service.refresh_index()

    judge_chat = LLMJudge(chat, max_tokens=400)

    # 优先用跨厂商裁判；没配或连不通就退回同厂商 R1（结论只能作参考）
    cross = build_judge_llm(settings)
    if cross is not None:
        ok, reason = await probe_client(cross)
        if ok:
            judge_b = LLMJudge(cross, max_tokens=800)
            label_b = f"{cross.name}（跨厂商）"
        else:
            print(f"⚠ 跨厂商裁判不可用：{reason[:120]}")
            print("  检查 JUDGE_API_KEY / JUDGE_BASE_URL / JUDGE_MODEL，本次回退到同厂商 R1\n")
            cross = None
    if cross is None:
        judge_b = LLMJudge(reasoner, max_tokens=1600)
        label_b = "deepseek-reasoner（同厂商，参考值）"

    print(f"主裁判：deepseek-chat   副裁判：{label_b}")
    cases = load_cases(BASE / "eval" / "qa_set.jsonl")[: args.limit]
    print(f"复核样本：{len(cases)} 条；每条 = 1 次生成 + 4 次裁判（双序各两次）…\n")

    rule_rates: list[float] = []
    grounded_chat: list[float] = []
    grounded_b: list[float] = []
    ok_chat_b = 0
    compared = 0
    order_sensitive = 0
    order_deltas: list[int] = []
    disagreements: list[str] = []
    rule_issues: list[str] = []

    for i, case in enumerate(cases, start=1):
        answer, sources = await collect(service, case.question)

        rule = verify_answer(answer, sources)
        if rule["rate"] is not None:
            rule_rates.append(rule["rate"])
        for issue in rule["issues"]:
            rule_issues.append(
                f"{case.question} → 角标[{issue['cite']}] {issue['reason']}：{issue['claim'][:50]}"
            )

        v_chat = await judge_robust(judge_chat, case.question, sources, answer)
        v_b = await judge_robust(judge_b, case.question, sources, answer)

        if v_chat:
            grounded_chat.append(v_chat.grounded)
            if v_chat.order_delta > 0:
                order_sensitive += 1
                order_deltas.append(v_chat.order_delta)
        if v_b:
            grounded_b.append(v_b.grounded)
            if v_chat:
                compared += 1
                if round(v_b.grounded) == round(v_chat.grounded):
                    ok_chat_b += 1
                else:
                    disagreements.append(
                        f"{case.question}\n   chat={v_chat.grounded:.1f} 副裁判={v_b.grounded:.1f}"
                        f"  chat理由={v_chat.forward.reason[:50]}"
                    )

        def fmt(v) -> str:
            return f"{v.grounded:.1f}" if v else "-"

        print(
            f"  {i}/{len(cases)} rule={rule['rate']:.0%}"
            f" chat={fmt(v_chat)} 副裁判={fmt(v_b)}"
            f" 位置差={v_chat.order_delta if v_chat else '-'}"
            if rule["rate"] is not None
            else f"  {i}/{len(cases)} rule=n/a"
        )

    def pct(n: int, d: int) -> str:
        return f"{n / d:.0%}（{n}/{d}）" if d else "n/a"

    print("\n=== 复核结果（双序平均） ===")
    print(f"规则校验通过率     : {statistics.fmean(rule_rates):.1%}" if rule_rates else "规则校验: n/a")
    print(f"裁判 chat 均分     : {statistics.fmean(grounded_chat):.2f}" if grounded_chat else "")
    print(f"副裁判（{label_b}） : {statistics.fmean(grounded_b):.2f}" if grounded_b else "")
    print(f"chat↔副裁判 一致率  : {pct(ok_chat_b, compared)}")
    if grounded_chat:
        ratio = order_sensitive / len(grounded_chat)
        avg_delta = statistics.fmean(order_deltas) if order_deltas else 0.0
        print(f"位置敏感样本       : {order_sensitive}/{len(grounded_chat)}（{ratio:.0%}，平均差 {avg_delta:.2f} 分）")

    if rule_issues:
        print("\n规则校验发现的问题（最可能是真错误）：")
        for item in rule_issues[:10]:
            print(f"  - {item}")
    if disagreements:
        print("\n两个裁判打分不一致：")
        for item in disagreements[:8]:
            print(f"  - {item}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
