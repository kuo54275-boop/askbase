"""参数消融实验。

在内存里重建索引，不改数据库、不调大模型，几十毫秒跑完一组配置。

用法：
    python scripts/ablation.py            # 输出 markdown 表格
    python scripts/ablation.py --json     # 输出 JSON

回答两个面试必问的问题：
1. 「混合检索真的比单路好吗？」—— 有对照组，有数字。
2. 「切分参数怎么定的？」—— 有网格搜索，不是拍脑袋。
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from app.config import get_settings  # noqa: E402
from app.evaluation import load_cases  # noqa: E402
from app.ingest.chunker import chunk_text, index_text  # noqa: E402
from app.ingest.loaders import extract_text  # noqa: E402
from app.retrieval.hybrid import HybridRetriever  # noqa: E402

KS = (1, 3, 5)


def build_index(chunk_size: int, overlap: int):
    """按给定切分参数重建内存索引（不落库）。"""
    settings = get_settings()
    items: list[tuple[str, str]] = []
    doc_of: dict[str, str] = {}

    for path in sorted(settings.samples_dir.iterdir()):
        if not path.is_file():
            continue
        text = extract_text(path.name, path.read_bytes())
        for piece in chunk_text(text, chunk_size, overlap):
            cid = f"{path.name}-{piece.ordinal}"
            items.append((cid, index_text(piece.section, piece.text)))
            doc_of[cid] = path.name

    retriever = HybridRetriever(rrf_k=60, embed_provider=settings.embedding_provider)
    retriever.build(items)
    return retriever, doc_of


def run(cases, retriever, doc_of, top_k: int = 5, **search_kwargs) -> dict:
    hits_at = {k: 0 for k in KS}
    rr: list[float] = []
    latencies: list[float] = []

    for case in cases:
        started = time.perf_counter()
        hits = retriever.search(case.question, top_k=max(KS), **search_kwargs)
        latencies.append((time.perf_counter() - started) * 1000)

        names = [doc_of.get(cid) for cid, _ in hits]
        rank = next((i for i, n in enumerate(names, start=1) if n == case.gold_doc), None)
        for k in KS:
            if rank and rank <= k:
                hits_at[k] += 1
        rr.append(1.0 / rank if rank else 0.0)

    total = len(cases) or 1
    return {
        "n": len(cases),
        "recall": {f"@{k}": round(hits_at[k] / total, 4) for k in KS},
        "mrr": round(statistics.fmean(rr), 4) if rr else 0.0,
        "latency_ms": round(statistics.fmean(latencies), 3) if latencies else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    cases = load_cases(BASE / "eval" / "qa_set.jsonl")
    rows: list[dict] = []

    def add(group: str, label: str, retriever, doc_of, **kw) -> None:
        metrics = run(cases, retriever, doc_of, **kw)
        rows.append({"group": group, "label": label, **metrics})

    # ---------- 1. 召回通路消融 ----------
    ret, doc_of = build_index(400, 60)
    add("召回通路", "BM25 only", ret, doc_of, use_vector=False)
    add("召回通路", "向量 only", ret, doc_of, use_bm25=False)
    add("召回通路", "混合 + 门控", ret, doc_of)
    add("召回通路", "混合（关闭门控）", ret, doc_of, min_cos_ratio=0.0, bm25_min_ratio=0.0)

    # ---------- 2. 切分参数网格 ----------
    for size in (300, 400, 600):
        for overlap in (0, 60, 120):
            if overlap >= size // 2:
                continue
            ret, doc_of = build_index(size, overlap)
            add("切分参数", f"size={size}, overlap={overlap}", ret, doc_of)

    # ---------- 3. RRF 的 k ----------
    for k in (10, 60, 100):
        ret, doc_of = build_index(400, 60)
        ret.rrf_k = k
        add("RRF k", f"k={k}", ret, doc_of)

    # ---------- 4. 融合权重（应对"弱召回通路拖后腿"）----------
    for bm, vec in ((1.0, 1.0), (0.8, 0.4), (0.7, 0.3), (0.6, 0.2), (0.5, 0.1)):
        ret, doc_of = build_index(400, 60)
        add(
            "融合权重",
            f"bm25={bm}, vec={vec}",
            ret,
            doc_of,
            bm25_weight=bm,
            vector_weight=vec,
        )

    # ---------- 5. 门控强度 ----------
    for bm_ratio in (0.0, 0.5, 0.65, 0.8):
        ret, doc_of = build_index(400, 60)
        add(
            "门控强度",
            f"bm25_cut={bm_ratio}",
            ret,
            doc_of,
            bm25_weight=0.6,
            vector_weight=0.2,
            bm25_min_ratio=bm_ratio,
        )

    if args.json:
        print(json.dumps(rows, ensure_ascii=False, indent=2))
        return 0

    print("| 分组 | 配置 | Recall@1 | Recall@3 | Recall@5 | MRR | 延迟(ms) |")
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        print(
            f"| {r['group']} | {r['label']} | "
            f"{r['recall']['@1']:.2%} | {r['recall']['@3']:.2%} | {r['recall']['@5']:.2%} | "
            f"{r['mrr']:.4f} | {r['latency_ms']:.3f} |"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
