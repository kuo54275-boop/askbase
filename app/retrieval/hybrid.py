"""混合检索：BM25 + 向量，用 RRF 融合。

为什么不用「分数加权相加」：
BM25 分数是 (0, +∞) 的无界值，余弦是 [-1, 1]，两者量纲完全不同，
加权相加必须做归一化，而归一化方式（min-max / z-score）会引入超参，
换一批数据就得重调。

RRF（Reciprocal Rank Fusion）只看名次，天然对分数尺度免疫：
    score(d) = Σ_lists 1 / (k + rank_list(d))

k=60 是论文里的经验值，作用是压住 top-1 的过度主导。
"""

from __future__ import annotations

from .bm25 import BM25Index
from .vector import VectorIndex


class HybridRetriever:
    def __init__(self, rrf_k: int = 60, embed_provider: str = "hashing") -> None:
        self.rrf_k = rrf_k
        self.bm25 = BM25Index()
        self.vector = VectorIndex(_provider(embed_provider))
        self._indexed = False

    # ---------- 建索引 ----------
    def build(self, items: list[tuple[str, str]]) -> None:
        """items: [(chunk_id, index_text)]"""
        self.bm25 = BM25Index()
        for cid, text in items:
            self.bm25.add(cid, text)
        self.bm25.build()
        self.vector.build(items)
        self._indexed = True

    def __len__(self) -> int:
        return len(self.vector)

    # ---------- 检索 ----------
    def search(
        self,
        query: str,
        top_k: int = 5,
        candidates: int = 20,
        min_cos_ratio: float = 0.4,
        bm25_min_ratio: float = 0.5,
        use_bm25: bool = True,
        use_vector: bool = True,
        bm25_weight: float = 1.0,
        vector_weight: float = 1.0,
    ) -> list[tuple[str, float]]:
        """返回 [(chunk_id, fused_score)]，按融合分降序。

        两道门控，各管一路召回：

        1. min_cos_ratio（向量侧）：余弦相似度低于 top * 0.4 的切片直接丢弃。
           哈希向量对共享字符的文本会给出 0.02 这种"伪正相似"，不裁掉会污染 prompt。
        2. bm25_min_ratio（关键词侧）：BM25 分数低于 top * 0.5 的切片丢弃。
           BM25 的分数是连续信号，通常存在明显断层（如 19.8 vs 8.6），
           断层以下基本是"碰巧共享几个字"的噪声。

        两路各裁各的，再由 RRF 融合：即使某一路误裁了正确切片，
        另一路仍能把它召回来 —— 这正是混合检索比单路更稳的原因。
        """
        if not self._indexed or not query.strip():
            return []

        bm25_hits = self.bm25.search(query, top_k=candidates) if use_bm25 else []
        if bm25_hits:
            floor = bm25_hits[0][1] * bm25_min_ratio
            bm25_hits = [(cid, s) for cid, s in bm25_hits if s >= floor]

        vec_hits = self.vector.search(query, top_k=candidates) if use_vector else []
        if vec_hits:
            # 相对门控（跟随本 query 的最高分）+ 绝对门控（该向量表示的经验下限）
            abs_floor = getattr(self.vector.provider, "min_similarity", 0.0)
            floor = max(vec_hits[0][1] * min_cos_ratio, abs_floor)
            vec_hits = [(cid, s) for cid, s in vec_hits if s >= floor]

        fused: dict[str, float] = {}
        for hits, weight in ((bm25_hits, bm25_weight), (vec_hits, vector_weight)):
            for rank, (cid, _score) in enumerate(hits, start=1):
                fused[cid] = fused.get(cid, 0.0) + weight / (self.rrf_k + rank)

        ranked = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
        return ranked[:top_k]


def _provider(name: str):
    from .vector import get_embedding_provider

    return get_embedding_provider(name)
