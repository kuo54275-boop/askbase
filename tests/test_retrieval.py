"""检索层单测：BM25、向量、RRF 融合。"""

from __future__ import annotations

from app.retrieval.bm25 import BM25Index
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.vector import HashingEmbedding, VectorIndex, cosine
from app.retrieval.text import tokenize


def test_tokenize_mixes_cjk_and_ascii():
    tokens = tokenize("报销 Process 2026")
    assert "报" in tokens
    assert "报销" in tokens
    assert "process" in tokens


def test_bm25_ranks_matching_doc_first():
    idx = BM25Index()
    idx.add("a", "报销需要提交发票和消费明细小票")
    idx.add("b", "年假申请需要提前三个工作日提交")
    idx.add("c", "公司提供弹性工作制")
    idx.build()

    hits = idx.search("报销要什么材料", top_k=3)
    assert hits[0][0] == "a"
    assert hits[0][1] > 0


def test_bm25_empty_index_returns_nothing():
    idx = BM25Index()
    idx.build()
    assert idx.search("任意查询") == []


def test_cosine_is_bounded():
    emb = HashingEmbedding()
    a = emb.embed("报销流程说明")
    b = emb.embed("报销流程说明")
    assert cosine(a, b) > 0.99
    c = emb.embed("完全不相关的内容 xyz")
    assert -1.0 <= cosine(a, c) <= 1.0


def test_vector_index_returns_positive_scores_only():
    vi = VectorIndex(HashingEmbedding())
    vi.build([("1", "报销流程"), ("2", "年假制度")])
    hits = vi.search("报销", top_k=5)
    assert hits and hits[0][0] == "1"
    assert all(score > 0 for _, score in hits)


def test_rrf_fuses_both_rankers():
    retriever = HybridRetriever(rrf_k=60)
    retriever.build(
        [
            ("d1", "报销流程：提交发票、消费明细，主管审批后财务打款"),
            ("d2", "年假制度：入职满一年享五天带薪年假"),
            ("d3", "差旅标准：一线城市住宿每晚不超过六百元"),
        ]
    )
    hits = retriever.search("报销需要发票吗", top_k=2)
    assert hits[0][0] == "d1"


def test_empty_query_returns_empty():
    retriever = HybridRetriever()
    retriever.build([("d1", "任意内容")])
    assert retriever.search("   ") == []
