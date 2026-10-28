"""BM25 关键词检索（自己实现，不依赖 rank_bm25）。

BM25 解决的是"词频饱和 + 文档长度归一"问题：
    score(q, d) = Σ IDF(t) * (tf * (k1+1)) / (tf + k1 * (1 - b + b * |d|/avgdl))

优点：精确匹配强、可解释、无模型、毫秒级。
缺点：不认同义词和语义相近，所以要和向量检索做互补。
"""

from __future__ import annotations

import math
from collections import Counter

from .text import tokenize


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.doc_ids: list[str] = []
        self._term_freqs: list[Counter[str]] = []
        self._doc_len: list[int] = []
        self._doc_freq: Counter[str] = Counter()
        self._avgdl: float = 0.0
        self._n_docs: int = 0

    def add(self, doc_id: str, text: str) -> None:
        tokens = tokenize(text)
        tf = Counter(tokens)
        self.doc_ids.append(doc_id)
        self._term_freqs.append(tf)
        self._doc_len.append(len(tokens))
        for term in tf:
            self._doc_freq[term] += 1

    def build(self) -> None:
        self._n_docs = len(self.doc_ids)
        total = sum(self._doc_len)
        self._avgdl = total / self._n_docs if self._n_docs else 0.0

    def _idf(self, term: str) -> float:
        df = self._doc_freq.get(term, 0)
        if df == 0:
            return 0.0
        # 加 1 平滑，保证 IDF 恒为正，避免常见词被算成负分
        return math.log(1.0 + (self._n_docs - df + 0.5) / (df + 0.5))

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        if not self._n_docs:
            return []
        q_terms = [t for t in tokenize(query) if t in self._doc_freq]
        if not q_terms:
            return []

        scores: dict[int, float] = {}
        for term in q_terms:
            idf = self._idf(term)
            for idx, tf in enumerate(self._term_freqs):
                freq = tf.get(term)
                if not freq:
                    continue
                norm = 1.0 - self.b + self.b * (self._doc_len[idx] / (self._avgdl or 1.0))
                scores[idx] = scores.get(idx, 0.0) + idf * (freq * (self.k1 + 1)) / (freq + self.k1 * norm)

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        return [(self.doc_ids[i], s) for i, s in ranked]
