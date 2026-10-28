"""LLM 层单测：Mock 的关键词抽取与问题定位。"""

from __future__ import annotations

from app.llm import MockClient, _extract_keywords, _last_user_content, _normalize_cn_numbers


def test_extract_keywords_splits_cjk_into_bigrams():
    kws = _extract_keywords("年假有几天？")
    assert "年假" in kws
    assert "几天" in kws


def test_extract_keywords_keeps_short_labels():
    # P1 / 4G 这类短标识必须保留，否则会匹配到错误的段落
    assert "P1" in _extract_keywords("P1 工单多久响应")
    assert "4G" in _extract_keywords("4G 网络支持吗")


def test_normalize_cn_numbers_keeps_unit():
    assert _normalize_cn_numbers("三十万") == "30万"
    assert _normalize_cn_numbers("五千") == "5000"
    assert _normalize_cn_numbers("十五") == "15"


def test_last_user_content_extracts_question_from_rag_prompt():
    messages = [
        {"role": "system", "content": "系统提示"},
        {
            "role": "user",
            "content": "【资料】\n[1] 《手册》·年假\n满一年 5 天。\n\n【问题】\n年假有几天？\n\n请遵守以下规则：\n1. 只依据资料",
        },
    ]
    assert _last_user_content(messages) == "年假有几天？"


def test_mock_compose_picks_numeric_sentence():
    refs = [
        "员工入职满 1 年享有 5 天带薪年假，满 3 年享有 10 天，满 5 年享有 15 天。"
        "年假以自然年为单位计算，当年未休完的年假最多可结转 5 天，逾期作废。"
    ]
    answer = MockClient()._compose("年假有几天？", refs)
    assert "5 天带薪年假" in answer
    assert "[1]" in answer


def test_mock_compose_without_refs():
    answer = MockClient()._compose("随便问问", [])
    assert "没有检索到" in answer
