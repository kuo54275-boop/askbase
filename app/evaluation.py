"""离线评测。

秋招项目最容易翻车的地方：只讲"效果不错"，拿不出数字。
这里把评测做成可复现的脚本 + 固定标注集：

- Recall@k：标准答案所在文档是否出现在前 k 条检索结果里（衡量检索层）
- MRR：正确文档首次出现的排名倒数的均值（衡量排序质量）
- 关键词命中率：答案关键词是否被召回切片覆盖（无 LLM 也能算，衡量召回完整性）

评测只看检索层，不调用大模型 —— 所以无网、无 Key、零成本，可以进 CI。
"""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

from .models import EvalCase, EvalResult
from .rag import RagService


def load_cases(path: Path) -> list[EvalCase]:
    cases: list[EvalCase] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            cases.append(EvalCase(**json.loads(line)))
    return cases


def evaluate(service: RagService, cases: list[EvalCase], ks: tuple[int, ...] = (1, 3, 5)) -> EvalResult:
    max_k = max(ks)
    hits_at: dict[int, int] = {k: 0 for k in ks}
    reciprocal_ranks: list[float] = []
    keyword_hits = 0
    latencies: list[float] = []
    details: list[dict] = []

    for case in cases:
        started = time.perf_counter()
        sources = service.retrieve(case.question, top_k=max_k)
        latencies.append((time.perf_counter() - started) * 1000)

        doc_names = [s.doc_name for s in sources]
        first_rank = None
        for rank, name in enumerate(doc_names, start=1):
            if name == case.gold_doc:
                first_rank = rank
                break

        for k in ks:
            if first_rank is not None and first_rank <= k:
                hits_at[k] += 1
        reciprocal_ranks.append(1.0 / first_rank if first_rank else 0.0)

        joined = "\n".join(s.text for s in sources)
        kw_ok = bool(case.answer_keywords) and all(kw in joined for kw in case.answer_keywords)
        if kw_ok:
            keyword_hits += 1

        details.append(
            {
                "question": case.question,
                "gold_doc": case.gold_doc,
                "first_hit_rank": first_rank,
                "keywords_ok": kw_ok,
                "top_docs": doc_names[: max(ks)],
            }
        )

    total = len(cases) or 1
    return EvalResult(
        total=len(cases),
        recall_at={f"@{k}": round(hits_at[k] / total, 4) for k in ks},
        mrr=round(statistics.fmean(reciprocal_ranks), 4) if reciprocal_ranks else 0.0,
        avg_latency_ms=round(statistics.fmean(latencies), 2) if latencies else 0.0,
        details=details,
    )
