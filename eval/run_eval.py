"""离线评测 CLI。

用法：
    python -m eval.run_eval
    python -m eval.run_eval --json     # 输出 JSON，方便贴进 README

只测检索层，不调用大模型，零 token 成本。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from app.config import get_settings  # noqa: E402
from app.db import Repository  # noqa: E402
from app.evaluation import evaluate, load_cases  # noqa: E402
from app.llm import build_llm  # noqa: E402
from app.rag import RagService  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true", help="只输出 JSON")
    parser.add_argument("--cases", default=str(BASE / "eval" / "qa_set.jsonl"))
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
    service = RagService(settings, repo, llm)
    service.refresh_index()

    cases = load_cases(Path(args.cases))
    result = evaluate(service, cases)

    if args.json:
        print(json.dumps(json.loads(result.model_dump_json()), ensure_ascii=False, indent=2))
        return 0

    print(f"样本数：{result.total}")
    for k, v in result.recall_at.items():
        print(f"Recall{k}: {v:.2%}")
    print(f"MRR: {result.mrr:.4f}")
    print(f"平均检索延迟: {result.avg_latency_ms:.2f} ms")
    print("\n未命中样本：")
    misses = [d for d in result.details if d["first_hit_rank"] is None]
    if not misses:
        print("  无")
    for d in misses:
        print(f"  - {d['question']} (期望: {d['gold_doc']}, 实际: {d['top_docs'][:3]})")

    # 排在第 2、3 位的样本同样值得看：它们决定了"多给几条上下文能不能救回来"
    later = [d for d in result.details if (d["first_hit_rank"] or 0) > 1]
    print(f"\n非首位命中（共 {len(later)} 条，Recall@3 内可救回）：")
    if not later:
        print("  无")
    for d in later:
        picked = [n for n in d["top_docs"][:3]]
        print(f"  - #{d['first_hit_rank']} {d['question']}")
        print(f"      期望 {d['gold_doc']} / 实际 {picked}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
