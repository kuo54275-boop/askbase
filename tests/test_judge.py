"""裁判层单测：JSON 解析容错、引用抽取、拒答识别。"""

from __future__ import annotations

from app.judge import extract_citations, is_refusal, parse_judge_output


def test_parse_plain_json():
    r = parse_judge_output('{"grounded": 2, "relevant": 2, "bad_citations": [], "reason": "ok"}')
    assert r is not None
    assert r.grounded == 2
    assert r.relevant == 2
    assert r.bad_citations == []


def test_parse_json_in_code_fence():
    raw = '```json\n{"grounded": 1, "relevant": 2, "bad_citations": [2], "reason": "x"}\n```'
    r = parse_judge_output(raw)
    assert r is not None
    assert r.grounded == 1
    assert r.bad_citations == [2]


def test_parse_json_with_chatter():
    raw = '好的，我的判断是：{"grounded": 0, "relevant": 1, "bad_citations": [1, 3]} 以上。'
    r = parse_judge_output(raw)
    assert r is not None
    assert r.grounded == 0
    assert r.bad_citations == [1, 3]


def test_parse_rejects_invalid_score():
    assert parse_judge_output('{"grounded": 5, "relevant": 2}') is None
    assert parse_judge_output("这不是 JSON") is None


def test_parse_tolerates_string_citations():
    r = parse_judge_output('{"grounded": 2, "relevant": 2, "bad_citations": ["2", "x"]}')
    assert r is not None
    assert r.bad_citations == [2]


def test_extract_citations_dedup_and_order():
    assert extract_citations("结论 A[2]，结论 B[1]，补充[2]。") == [2, 1]
    assert extract_citations("没有引用") == []


def test_is_refusal_variants():
    assert is_refusal("资料中未提及该公司股票代码。")
    assert is_refusal("根据现有资料，无法确定食堂开饭时间。")
    assert is_refusal("资料中没有相关信息。")
    assert not is_refusal("报销需要提交发票和消费明细。[1]")
    assert not is_refusal("")
