"""大模型客户端。

两个实现，同一个接口：
- DeepSeekClient：真实调用，SSE 流式输出，带重试和超时；
- MockClient：没有 API Key 时的降级方案，从检索结果里抽取句子拼答案。

Mock 不是玩具：它保证「pip 装完、克隆下来就能演示」，评测脚本也能在无网无 key 时跑通。
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import AsyncIterator, Protocol

import httpx

SYSTEM_PROMPT = (
    "你是一名严谨的企业知识库助手。你只能依据用户提供的【资料】回答，"
    "不得使用资料以外的知识，也不得编造事实。"
)


class LLMError(Exception):
    """调用大模型失败（网络、鉴权、限流等）。"""


class LLMClient(Protocol):
    name: str

    async def stream(
        self,
        messages: list[dict],
        refs: list[str] | None = None,
    ) -> AsyncIterator[str]: ...

    async def complete(
        self, messages: list[dict], max_tokens: int = 80, json_mode: bool = False
    ) -> str: ...
class DeepSeekClient:
    """DeepSeek 官方 API（OpenAI 兼容协议）。"""

    name = "deepseek"

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.deepseek.com",
        model: str = "deepseek-chat",
        timeout: float = 60.0,
        max_retries: int = 2,
        name: str = "deepseek",
    ) -> None:
        if not api_key:
            raise LLMError("API Key 未配置")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        # 同一个客户端可以指向任意 OpenAI 兼容端点（通义/Kimi/智谱/OpenAI…），
        # name 只用于日志和 /healthz 展示
        self.name = name

    async def stream(
        self,
        messages: list[dict],
        refs: list[str] | None = None,
    ) -> AsyncIterator[str]:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "temperature": 0.2,
            "max_tokens": 900,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            emitted = False
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    async with client.stream(
                        "POST",
                        f"{self.base_url}/chat/completions",
                        json=payload,
                        headers=headers,
                    ) as resp:
                        if resp.status_code >= 400:
                            body = (await resp.aread()).decode("utf-8", "ignore")[:300]
                            raise LLMError(f"{self.name} 返回 {resp.status_code}: {body}")

                        async for line in resp.aiter_lines():
                            if not line or not line.startswith("data:"):
                                continue
                            data = line[5:].strip()
                            if data == "[DONE]":
                                return
                            try:
                                obj = json.loads(data)
                            except json.JSONDecodeError:
                                continue
                            choices = obj.get("choices") or []
                            if not choices:
                                continue
                            delta = choices[0].get("delta") or {}
                            piece = delta.get("content")
                            if piece:
                                emitted = True
                                yield piece
                return
            except (httpx.TransportError, httpx.TimeoutException, LLMError) as exc:
                last_error = exc
                # 已经吐出部分 token 就不要重试，否则前端会看到重复内容
                if emitted or attempt >= self.max_retries:
                    break
                await asyncio.sleep(0.8 * (2**attempt))

        raise LLMError(f"大模型调用失败：{last_error}")

    async def complete(
        self, messages: list[dict], max_tokens: int = 80, json_mode: bool = False
    ) -> str:
        """非流式短调用：query 改写、重排打分、评测裁判这类"只要一小段文本"的场景。"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload: dict = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "temperature": 0.0,
            "max_tokens": max_tokens,
        }
        if json_mode:
            # DeepSeek 兼容 OpenAI 的 JSON 模式，能显著降低解析失败率
            payload["response_format"] = {"type": "json_object"}
        try:
            async with httpx.AsyncClient(timeout=min(self.timeout, 20.0)) as client:
                resp = await client.post(
                    f"{self.base_url}/chat/completions", json=payload, headers=headers
                )
                if resp.status_code >= 400:
                    # 带上响应体：invalid_api_key / model_not_found 这类原因必须能看见
                    raise LLMError(
                        f"{self.name} 返回 {resp.status_code}: {resp.text[:200]}"
                    )
                data = resp.json()
                return (data["choices"][0]["message"]["content"] or "").strip()
        except (httpx.HTTPError, KeyError, IndexError) as exc:
            raise LLMError(f"改写调用失败：{exc}") from exc


class MockClient:
    """无 Key 降级：抽取式回答，保证链路可演示、可测试。"""

    name = "mock"

    def __init__(self) -> None:
        self._note = "（Mock 模式：未配置 DEEPSEEK_API_KEY，以下内容由检索结果抽取生成）\n\n"

    async def stream(
        self,
        messages: list[dict],
        refs: list[str] | None = None,
    ) -> AsyncIterator[str]:
        question = _last_user_content(messages)
        refs = refs or []
        answer = self._compose(question, refs)
        # 模拟 token 流式输出，前端体验和真实调用一致
        for para in answer.split("\n\n"):
            for i in range(0, len(para), 12):
                yield para[i : i + 12]
                await asyncio.sleep(0.012)
            yield "\n\n"

    async def complete(
        self, messages: list[dict], max_tokens: int = 80, json_mode: bool = False
    ) -> str:
        # Mock 不做真实改写，返回空串让上层走启发式兜底（离线也有多轮能力）
        return ""

    def _compose(self, question: str, refs: list[str]) -> str:
        if not refs:
            return self._note + "资料库中没有检索到与问题相关的内容，请先上传文档。"

        keywords = _extract_keywords(question)
        # 问"多少/几/多久"时，优先选带数字的句子——Mock 没有语义能力，靠这个规则补齐
        wants_number = bool(re.search(r"多少|几|多久|多长|上限|标准|时间", question))
        picked: list[str] = []

        for idx, ref in enumerate(refs, start=1):
            sentences = [s.strip() for s in re.split(r"(?<=[。！？；])", ref) if s.strip()]
            if not sentences:
                continue

            def score(sent: str) -> tuple[int, int]:
                # 去空格后再匹配，避免「30 万元」和「30万」对不上
                compact_sent = re.sub(r"\s+", "", sent)
                hit = sum(1 for k in keywords if k in compact_sent)
                # 问数量时，含数字越多的句子越可能是答案（如「满 1 年 5 天、满 3 年 10 天」）
                nums = len(re.findall(r"\d+", compact_sent)) if wants_number else 0
                return (hit, nums)

            best = max(sentences, key=score)
            if score(best)[0] > 0:
                picked.append(f"- {best} [{idx}]")
            if len(picked) >= 3:
                break

        if not picked:
            lead = (refs[0].split("。")[0] or refs[0][:120]).strip()
            picked = [f"- {lead} [1]"]

        return (
            self._note
            + "根据知识库检索结果：\n\n"
            + "\n".join(picked)
            + "\n\n（如需完整生成式回答，请配置 DEEPSEEK_API_KEY）"
        )


_CN_DIGITS = {"零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
_CN_UNITS = {"十": 10, "百": 100, "千": 1000}


def _normalize_cn_numbers(text: str) -> str:
    """把「三十万」「五千」这类中文数字转成阿拉伯数字。

    文档里写的是「30 万元」，用户问的是「三十万」——不归一化就永远匹配不上。
    只处理到万位，够用了。
    """

    def convert(match: re.Match) -> str:
        token = match.group(0)
        suffix = ""
        if token.endswith(("万", "亿")):
            suffix = token[-1]
            token = token[:-1]

        total, current = 0, 0
        for ch in token:
            if ch in _CN_DIGITS:
                current = _CN_DIGITS[ch]
            elif ch in _CN_UNITS:
                total += (current or 1) * _CN_UNITS[ch]
                current = 0
            else:
                return match.group(0)
        value = total + current
        if value == 0:
            return match.group(0)
        # 保留「万/亿」单位，让「三十万」变成「30万」，才能匹配文档里的「30 万元」
        if suffix:
            return f"{value}{suffix}"
        return str(value)

    return re.sub(r"[零一二两三四五六七八九十百千]+[万亿]?", convert, text)


def _extract_keywords(text: str) -> list[str]:
    """抽取用于 Mock 匹配的关键词。

    两个坑：
    1. 中文整段当成一个 token 会导致「一句话都匹配不上」，必须切成 bigram；
    2. 漏掉 P1 / 4G 这类"字母+数字"短标识，会导致问「P1 多久响应」时匹配到 P3 的条款。
    """
    compact = re.sub(r"\s+", "", _normalize_cn_numbers(text))
    found: list[str] = []

    for run in re.findall(r"[\u4e00-\u9fff]+", compact):
        if len(run) == 1:
            found.append(run)
        else:
            found.extend(run[i : i + 2] for i in range(len(run) - 1))

    found.extend(
        re.findall(
            r"[A-Za-z]{1,3}\d{1,3}|\d{1,3}[A-Za-z]{1,3}|[A-Za-z]{3,}|\d+",
            compact,
        )
    )

    seen: set[str] = set()
    result: list[str] = []
    for word in found:
        if len(word) < 2 or word in seen:
            continue
        seen.add(word)
        result.append(word)
    return result


def _last_user_content(messages: list[dict]) -> str:
    """取出用户真正问的那句话。

    注意：RAG 的 user message 是「【资料】…【问题】…规则」拼成的一大段，
    如果直接把整段当问题，Mock 的关键词会从资料正文里抽，匹配结果全乱。
    这个 bug 单测（直接传问题）测不出来，只有端到端才会暴露。
    """
    for msg in reversed(messages):
        if msg.get("role") != "user":
            continue
        content = str(msg.get("content", ""))
        if "【问题】" in content:
            question = content.split("【问题】", 1)[1]
            question = question.split("请遵守以下规则", 1)[0]
            return question.strip()
        return content
    return ""


def build_llm(
    provider: str,
    api_key: str,
    base_url: str,
    model: str,
    timeout: float,
    max_retries: int,
) -> LLMClient:
    """根据配置构造客户端；provider 已经是解析过的真实值（deepseek / mock）。"""
    if provider == "deepseek":
        return DeepSeekClient(api_key, base_url, model, timeout, max_retries)
    return MockClient()


def build_judge_llm(settings) -> LLMClient | None:
    """构造跨厂商裁判客户端；没配置就返回 None，由调用方回退到主模型。

    任何兼容 OpenAI Chat Completions 协议的服务都能用：
      通义千问  https://dashscope.aliyuncs.com/compatible-mode/v1
      Kimi      https://api.moonshot.cn/v1
      智谱 GLM  https://open.bigmodel.cn/api/paas/v4
      硅基流动   https://api.siliconflow.cn/v1
      OpenAI    https://api.openai.com/v1
    """
    if not (settings.judge_api_key and settings.judge_base_url and settings.judge_model):
        return None
    return DeepSeekClient(
        api_key=settings.judge_api_key,
        base_url=settings.judge_base_url,
        model=settings.judge_model,
        timeout=settings.llm_timeout,
        max_retries=settings.llm_max_retries,
        name=f"judge:{settings.judge_model}",
    )


async def probe_client(client: LLMClient) -> tuple[bool, str]:
    """用一次最小调用探活。

    跨厂商配置最常见的问题是 key 复制错、没开通服务、地域不匹配。
    这些错误不该让整个评测脚本崩掉，而应该给出可读的失败原因并回退。
    """
    try:
        await client.complete([{"role": "user", "content": "ping"}], max_tokens=5)
        return True, ""
    except LLMError as exc:
        return False, str(exc)
    except Exception as exc:  # 网络、DNS、证书等
        return False, f"{type(exc).__name__}: {exc}"
