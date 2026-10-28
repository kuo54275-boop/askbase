"""裁判探活单测：坏配置必须降级而不是崩溃。"""

from __future__ import annotations

import asyncio

from app.llm import LLMError, MockClient, probe_client


class _Boom:
    name = "judge:broken"

    async def complete(self, messages, max_tokens: int = 80, json_mode: bool = False) -> str:
        raise LLMError("judge:broken 返回 401: invalid_api_key")

    async def stream(self, messages, refs=None):  # pragma: no cover
        yield ""


class _Down:
    name = "judge:down"

    async def complete(self, messages, max_tokens: int = 80, json_mode: bool = False) -> str:
        raise ConnectionError("DNS 解析失败")

    async def stream(self, messages, refs=None):  # pragma: no cover
        yield ""


def test_probe_ok():
    ok, reason = asyncio.run(probe_client(MockClient()))
    assert ok is True
    assert reason == ""


def test_probe_reports_auth_failure():
    ok, reason = asyncio.run(probe_client(_Boom()))
    assert ok is False
    assert "invalid_api_key" in reason


def test_probe_reports_network_failure():
    ok, reason = asyncio.run(probe_client(_Down()))
    assert ok is False
    assert "ConnectionError" in reason or "DNS" in reason
