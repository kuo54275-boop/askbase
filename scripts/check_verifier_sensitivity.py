"""检验规则校验器有没有"检出能力"。

一个永远判 100% 的检查器是没有价值的。这里人为把答案里的引用编号改错
（[1] → [2]），看规则能不能抓出来：

- 检出率高：说明规则真的在校验引用内容，之前的 100% 有意义；
- 检出率低：说明规则形同虚设，应该换判据。

用法：
    python scripts/check_verifier_sensitivity.py --limit 10
"""

from __future__ import annotations

import argparse
import asyncio
import re
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from app.config import get_settings  # noqa: E402
from app.db import Repository  # noqa: E402
from app.evaluation import load_cases  # noqa: E402
from app.llm import build_llm  # noqa: E402
from app.models import Source  # noqa: E402
from app.rag import RagService  # noqa: E402
from app.verify import verify_answer  # noqa: E402

_CITE_RE = re.compile(r"\[(\d{1,2})\]")


def corrupt(answer: str, n_sources: int) -> str:
    """把每个角标换成"下一份"资料，制造引用错配。"""

    def swap(match: re.Match) -> str:
        idx = int(match.group(1))
        return f"[{idx % n_sources + 1}]"

    return _CITE_RE.sub(swap, answer)


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

    cases = load_cases(BASE / "eval" / "qa_set.jsonl")[: args.limit]
    tested = 0
    detected = 0

    for case in cases:
        answer, sources = await collect(service, case.question)
        if len(sources) < 2 or not _CITE_RE.search(answer):
            continue  # 只有一份资料时无法制造错配

        bad_answer = corrupt(answer, len(sources))
        if bad_answer == answer:
            continue
        tested += 1

        clean = verify_answer(answer, sources)
        dirty = verify_answer(bad_answer, sources)
        caught = dirty["passed"] < clean["passed"] or len(dirty["issues"]) > len(clean["issues"])
        if caught:
            detected += 1
        print(
            f"  {'✓抓到' if caught else '✗漏过'}  原文通过 {clean['passed']}/{clean['checked']}"
            f" → 篡改后 {dirty['passed']}/{dirty['checked']}"
        )

    print(f"\n可测试样本：{tested} 条")
    if tested:
        print(f"篡改检出率：{detected / tested:.0%}（{detected}/{tested}）")
    else:
        print("没有可测试样本（大多是单来源答案）")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
