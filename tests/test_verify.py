"""引用校验单测：数字一致性、术语覆盖、异常引用。"""

from __future__ import annotations

from app.models import Source
from app.verify import (
    check_claim,
    is_meta_claim,
    is_negative_claim,
    is_unverifiable_claim,
    split_claims,
    verify_answer,
)


def _src(rank: int, text: str) -> Source:
    return Source(
        rank=rank, chunk_id=f"c{rank}", doc_name="doc.md", section="s", text=text, score=1.0
    )


def test_split_claims():
    assert len(split_claims("结论一。结论二！结论三？")) == 3


def test_number_must_appear_in_cited_source():
    ok, reason = check_claim(
        "响应时限为 15 分钟。[1]", ["P1：响应时限 15 分钟，解决时限 4 小时。"]
    )
    assert ok, reason


def test_missing_number_is_flagged():
    ok, reason = check_claim("赔付当月费用的 30%。[1]", ["低于 99.9% 的赔付 10%。"])
    assert not ok
    assert "30" in reason


def test_number_normalization_ignores_spaces():
    ok, _ = check_claim("默认保留 90 天。[1]", ["文档默认保留90天，可配置。"])
    assert ok


def test_term_coverage_without_numbers():
    ok, _ = check_claim(
        "由直属主管审批。[1]", ["年假申请需提前 3 个工作日在 OA 系统提交，由直属主管审批。"]
    )
    assert ok


def test_low_coverage_flagged():
    ok, reason = check_claim("需要董事会决议通过。[1]", ["报销需在费用发生后 30 天内提交。"])
    assert not ok
    assert "覆盖" in reason


def test_citation_to_missing_source():
    report = verify_answer("结论。[9]", [_src(1, "任意内容")])
    assert report["checked"] == 1
    assert report["passed"] == 0
    assert "不存在" in report["issues"][0]["reason"]


def test_verify_answer_aggregates():
    sources = [
        _src(1, "员工入职满 1 年享有 5 天带薪年假。"),
        _src(2, "报销需在费用发生后 30 天内提交。"),
    ]
    answer = "满 1 年享有 5 天年假。[1] 报销要在 30 天内提交。[2] 公司提供免费午餐。[1]"
    report = verify_answer(answer, sources)
    assert report["checked"] == 3
    assert report["passed"] == 2
    assert abs(report["rate"] - 2 / 3) < 1e-6


def test_meta_claim_is_skipped():
    """「依据：…上述…」这类指代总结句不是事实断言，不参与校验。"""
    sources = [_src(1, "私有化部署最低配置为 8 核 CPU、32GB 内存。")]
    answer = (
        "结论：私有化部署最低配置为 8 核 CPU、32GB 内存。[1] "
        "依据：《产品手册》中明确写明上述最低配置。[1]"
    )
    report = verify_answer(answer, sources)
    assert report["checked"] == 1
    assert report["passed"] == 1
    assert report["skipped"] == 1


def test_negative_claim_is_skipped():
    """「资料中未提及 X」是拒答，不是引用错误——不能算它不通过。"""
    sources = [_src(1, "拆分采购以规避审批权限的，视同越权审批。")]
    answer = "结论：资料中未提及把采购拆开来做的后果。[1]"
    report = verify_answer(answer, sources)
    assert report["checked"] == 0
    assert report["skipped"] == 1
    assert report["issues"] == []


def test_is_negative_claim():
    assert is_negative_claim("资料中未提及员工病假天数。")
    assert is_negative_claim("现有资料未说明试用期。")
    assert not is_negative_claim("最低配置为 8 核 CPU。")


def test_is_meta_claim():
    assert is_meta_claim("依据：资料[1]说明…")
    assert is_meta_claim("上述配置满足要求。")
    assert not is_meta_claim("最低配置为 8 核 CPU。")


def test_is_unverifiable_combines_both():
    assert is_unverifiable_claim("依据：上述配置…")
    assert is_unverifiable_claim("资料中未提及试用期。")
    assert not is_unverifiable_claim("报销需在 30 天内提交。[1]")


def test_answer_without_citations():
    report = verify_answer("没有角标的回答。", [_src(1, "x")])
    assert report["checked"] == 0
    assert report["rate"] is None


def test_multi_source_sentence_uses_union():
    """一句话引用两份资料时，按并集校验，避免误伤。"""
    sources = [
        _src(1, "文档默认保留 90 天。"),
        _src(2, "打印内容保留 24 小时后自动清除。"),
    ]
    answer = "文档默认保留 90 天；打印内容保留 24 小时后清除。[1][2]"
    report = verify_answer(answer, sources)
    assert report["checked"] == 2
    assert report["passed"] == 2
    assert report["issues"] == []


def test_union_does_not_mask_wrong_number():
    """并集校验不能变成"什么都放过"：编造的数字依然要被抓到。"""
    sources = [_src(1, "保留 90 天。"), _src(2, "保留 24 小时。")]
    report = verify_answer("保留 180 天后自动清除。[1][2]", sources)
    assert report["checked"] == 2
    assert report["passed"] == 0
    assert report["issues"]
