# 裁判接入单测：配置齐全才启用跨厂商，否则回退。

from __future__ import annotations

from app.config import Settings
from app.llm import build_judge_llm


def _settings(**kwargs) -> Settings:
    base = dict(
        _env_file=None,
        deepseek_api_key="sk-test",
        judge_api_key="",
        judge_base_url="",
        judge_model="",
    )
    base.update(kwargs)
    return Settings(**base)


def test_no_judge_config_returns_none():
    assert build_judge_llm(_settings()) is None


def test_partial_judge_config_returns_none():
    # 只填了 key，没填 base_url 和 model，不应该半启用
    assert build_judge_llm(_settings(judge_api_key="sk-other")) is None


def test_full_judge_config_builds_client():
    client = build_judge_llm(
        _settings(
            judge_api_key="sk-other",
            judge_base_url="https://api.moonshot.cn/v1",
            judge_model="moonshot-v1-8k",
        )
    )
    assert client is not None
    assert client.name == "judge:moonshot-v1-8k"
    assert client.base_url == "https://api.moonshot.cn/v1"
    assert client.model == "moonshot-v1-8k"


def test_judge_client_strips_trailing_slash():
    client = build_judge_llm(
        _settings(
            judge_api_key="sk-other",
            judge_base_url="https://api.openai.com/v1/",
            judge_model="gpt-4o-mini",
        )
    )
    assert client is not None
    assert client.base_url == "https://api.openai.com/v1"
