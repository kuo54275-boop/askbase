"""向量检索：零依赖的字符 n-gram 哈希向量。

为什么不直接上 bge / text-embedding：
- 中文 embedding 模型动辄几百 MB，还要 torch，秋招现场换台电脑就跑不起来；
- 哈希向量完全确定性、无网络、无模型，作为「可替换的基线」足够验证整条链路。

抽象成 EmbeddingProvider 接口，生产环境换成 bge-m3 / 外部 API 只改一个工厂函数，
检索层代码一行不用动 —— 这是面试时值得讲的分层设计。
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol

DIM = 1 << 14  # 16384 维，够用且内存友好

_ASCII_WORD = re.compile(r"[a-z0-9][a-z0-9_\-#+.]*")
_CJK_RUN = re.compile(r"[\u4e00-\u9fff]+")


class EmbeddingProvider(Protocol):
    name: str
    dim: int
    # 该向量表示下"有意义"的相似度下限，低于它的命中视为噪声
    min_similarity: float

    def embed(self, text: str) -> dict[int, float]: ...


def _features(text: str) -> list[str]:
    """字符级特征：中文 2-gram + 3-gram，英文整词。对错别字和分词错误更鲁棒。"""
    text = text.lower()
    feats = [f"w:{w}" for w in _ASCII_WORD.findall(text)]
    for run in _CJK_RUN.findall(text):
        feats.extend(f"c2:{run[i:i + 2]}" for i in range(len(run) - 1))
        feats.extend(f"c3:{run[i:i + 3]}" for i in range(len(run) - 2))
    return feats


class HashingEmbedding:
    """Signed hashing trick：把特征映射到固定维度并 L2 归一化。"""

    name = "hashing"
    dim = DIM
    # 哈希向量只会"共享字符"，相似度普遍偏低：实测相关段落约 0.10~0.15，
    # 不相关段落多在 0.02 以下，所以 0.05 是一个安全的经验下限。
    min_similarity = 0.05

    def embed(self, text: str) -> dict[int, float]:
        vec: dict[int, float] = {}
        for feat in _features(text):
            digest = hashlib.blake2b(feat.encode("utf-8"), digest_size=8).digest()
            raw = int.from_bytes(digest, "big")
            idx = raw % self.dim
            sign = 1.0 if (raw >> 63) & 1 else -1.0
            vec[idx] = vec.get(idx, 0.0) + sign

        norm = math.sqrt(sum(v * v for v in vec.values()))
        if norm == 0:
            return {}
        return {k: v / norm for k, v in vec.items()}


def cosine(a: dict[int, float], b: dict[int, float]) -> float:
    """入参已 L2 归一化，点积即余弦相似度。"""
    if not a or not b:
        return 0.0
    if len(a) > len(b):
        a, b = b, a
    return sum(v * b.get(k, 0.0) for k, v in a.items())


def get_embedding_provider(name: str) -> EmbeddingProvider:
    """工厂：后续接入 bge / 外部 embedding API 都从这里扩展。"""
    if name in ("hashing", "default"):
        return HashingEmbedding()
    raise ValueError(f"未知的 embedding provider: {name}")


class VectorIndex:
    """内存向量索引，全量暴力检索。"""

    def __init__(self, provider: EmbeddingProvider) -> None:
        self.provider = provider
        self._ids: list[str] = []
        self._vecs: list[dict[int, float]] = []

    def build(self, items: list[tuple[str, str]]) -> None:
        """items: [(chunk_id, index_text)]"""
        self._ids = [cid for cid, _ in items]
        self._vecs = [self.provider.embed(text) for _, text in items]

    def __len__(self) -> int:
        return len(self._ids)

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        q = self.provider.embed(query)
        if not q:
            return []
        scored = [(cid, cosine(q, vec)) for cid, vec in zip(self._ids, self._vecs)]
        scored = [(cid, s) for cid, s in scored if s > 0]
        scored.sort(key=lambda kv: kv[1], reverse=True)
        return scored[:top_k]
