"""BYOK 配置解析单测：用户填哪套配置，就解析成哪个 provider。"""

from __future__ import annotations

from app.config import Settings
from app.llm import build_llm_from_settings


def _s(**kwargs) -> Settings:
    base = dict(
        _env_file=None,
        llm_api_key="",
        llm_base_url="",
        llm_model="",
        deepseek_api_key="",
        llm_provider="auto",
    )
    base.update(kwargs)
    return Settings(**base)


def test_no_key_falls_back_to_mock():
    s = _s()
    assert s.effective_llm_provider == "mock"
    assert s.is_configured is False
    assert build_llm_from_settings(s).name == "mock"


def test_deepseek_key_only():
    s = _s(deepseek_api_key="sk-deepseek")
    assert s.effective_llm_provider == "deepseek"
    assert s.active_base_url == "https://api.deepseek.com"
    assert build_llm_from_settings(s).name == "deepseek"


def test_generic_openai_compatible_wins():
    s = _s(
        llm_api_key="sk-generic",
        llm_base_url="https://api.moonshot.cn/v1",
        llm_model="moonshot-v1-8k",
        deepseek_api_key="sk-deepseek",
    )
    assert s.effective_llm_provider == "openai"
    assert s.active_api_key == "sk-generic"
    assert s.active_model == "moonshot-v1-8k"
    client = build_llm_from_settings(s)
    assert client.name == "openai"
    assert client.base_url == "https://api.moonshot.cn/v1"


def test_partial_generic_config_is_ignored():
    # 只填了 key 没填 base_url，不应该半启用
    s = _s(llm_api_key="sk-generic", deepseek_api_key="")
    assert s.effective_llm_provider == "mock"


def test_base_url_trailing_slash_stripped():
    s = _s(llm_api_key="sk-x", llm_base_url="https://api.openai.com/v1/", llm_model="gpt-4o-mini")
    assert s.active_base_url == "https://api.openai.com/v1"
