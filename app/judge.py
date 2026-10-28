"""生成质量裁判（LLM as judge）。

检索指标只能回答"找没找对"，回答不了"答得对不对"。这里补三件事：

1. **忠实度（grounded）**：答案是否完全由检索到的资料支撑，有没有夹带模型自己的知识；
2. **引用准确率**：`[1]` 这种角标指向的资料，是否真的支持那句话；
3. **相关性（relevant）**：答案有没有答到点上。

裁判必须是**独立的一次调用**，且只看资料和答案、不看检索分数，
否则会倾向于"给自家系统打高分"。
"""

from __future__ import annotations

import json
import re

from .llm import LLMClient, LLMError
from .models import Source

_SYSTEM = (
    "你是一个严格、挑剔的 RAG 评测员。你的任务是判断【答案】是否完全基于【资料】，"
    "而不是判断答案是否符合常识或是否文笔好。\n"
    "评分标准：\n"
    "- grounded：2 = 每一条结论都能在资料里找到依据；"
    "1 = 大部分有依据，但夹带了资料之外的推断；0 = 明显编造或与资料矛盾。\n"
    "- relevant：2 = 直接回答了问题；1 = 沾边但没答到关键；0 = 答非所问。\n"
    "- bad_citations：列出「引用编号错误」的编号——即该编号对应的资料并不能支持那句话。"
    "如果答案没有引用角标，返回空数组。\n"
    '只输出 JSON，格式：{"grounded": 2, "relevant": 2, "bad_citations": [], "reason": "一句话"}'
)

_MAX_SOURCE_CHARS = 400

# 模型可能用各种说法表达"资料里没有"，这里尽量覆盖（只检查回答开头）
_REFUSAL_PATTERNS = [
    r"未提及",
    r"没有提到",
    r"未提到",
    r"资料中(并)?(没有|未|不)",
    r"未包含",
    r"没有(相关|包含)",
    r"无法(回答|确定|判断)",
    r"不确定",
    r"未说明",
    r"没有说明",
    r"无相关(信息|内容|资料)",
    r"未找到",
]


def is_refusal(answer: str) -> bool:
    """判断模型是否明确表示"资料里没有"。

    用来测拒答：知识库里不存在的问题，模型应该拒答而不是编造。
    """
    text = answer.strip()
    if not text:
        return False
    head = text[:200]
    return any(re.search(p, head) for p in _REFUSAL_PATTERNS)


class JudgeResult:
    __slots__ = ("grounded", "relevant", "bad_citations", "reason", "raw")

    def __init__(
        self,
        grounded: int,
        relevant: int,
        bad_citations: list[int],
        reason: str = "",
        raw: str = "",
    ) -> None:
        self.grounded = grounded
        self.relevant = relevant
        self.bad_citations = bad_citations
        self.reason = reason
        self.raw = raw


def extract_citations(answer: str) -> list[int]:
    """从答案里抽出 [1] [2] 这种角标编号（按出现顺序去重）。"""
    seen: list[int] = []
    for num in re.findall(r"\[(\d{1,2})\]", answer):
        idx = int(num)
        if idx not in seen:
            seen.append(idx)
    return seen


def parse_judge_output(raw: str) -> JudgeResult | None:
    """解析裁判输出，容忍 markdown 代码块和多余文字。"""
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?|\n?```$", "", text).strip()

    data = None
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.S)
        if match:
            try:
                data = json.loads(match.group(0))
            except json.JSONDecodeError:
                return None

    if not isinstance(data, dict):
        return None

    try:
        grounded = int(data.get("grounded", -1))
        relevant = int(data.get("relevant", -1))
    except (TypeError, ValueError):
        return None
    if grounded not in (0, 1, 2) or relevant not in (0, 1, 2):
        return None

    bad = data.get("bad_citations", [])
    bad_list: list[int] = []
    if isinstance(bad, list):
        for item in bad:
            try:
                bad_list.append(int(item))
            except (TypeError, ValueError):
                continue

    return JudgeResult(
        grounded=grounded,
        relevant=relevant,
        bad_citations=bad_list,
        reason=str(data.get("reason", ""))[:200],
        raw=raw,
    )


class LLMJudge:
    name = "llm-judge"

    def __init__(self, llm: LLMClient, max_tokens: int = 300) -> None:
        self.llm = llm
        # 推理型模型（如 deepseek-reasoner）的 token 预算要留给思维链，否则 content 为空
        self.max_tokens = max_tokens

    async def judge(self, question: str, sources: list[Source], answer: str) -> JudgeResult | None:
        if not answer.strip():
            return None

        context = "\n\n".join(
            f"[{s.rank}] 《{s.doc_name}》· {s.section}\n{s.text[:_MAX_SOURCE_CHARS]}"
            for s in sources
        ) or "（无资料）"

        messages = [
            {"role": "system", "content": _SYSTEM},
            {
                "role": "user",
                "content": (
                    f"【问题】\n{question}\n\n【资料】\n{context}\n\n【答案】\n{answer}"
                ),
            },
        ]

        try:
            raw = await self.llm.complete(messages, max_tokens=self.max_tokens, json_mode=True)
        except LLMError:
            return None
        return parse_judge_output(raw)


def reorder_sources(sources: list[Source], reverse: bool = False) -> list[Source]:
    """复制一份来源列表（可选倒序）并重新编号 1..n。

    倒序是检验位置偏差的手段：LLM 普遍偏爱靠前的材料，
    如果换个顺序分数就变，说明评分不稳定。
    """
    items = list(reversed(sources)) if reverse else list(sources)
    out: list[Source] = []
    for i, src in enumerate(items, start=1):
        out.append(
            Source(
                rank=i,
                chunk_id=src.chunk_id,
                doc_name=src.doc_name,
                section=src.section,
                text=src.text,
                score=src.score,
            )
        )
    return out


class RobustVerdict:
    """双序打分的合并结果。"""

    __slots__ = ("grounded", "relevant", "bad_citations", "order_delta", "forward", "reverse")

    def __init__(
        self,
        grounded: float,
        relevant: float,
        bad_citations: list[int],
        order_delta: int,
        forward: JudgeResult,
        reverse: JudgeResult,
    ) -> None:
        self.grounded = grounded
        self.relevant = relevant
        self.bad_citations = bad_citations
        self.order_delta = order_delta
        self.forward = forward
        self.reverse = reverse


async def judge_robust(
    judge: LLMJudge,
    question: str,
    sources: list[Source],
    answer: str,
) -> RobustVerdict | None:
    """正序 + 倒序各判一次，取平均，用来抵消位置偏差。

    两次打分都失败才返回 None；只有一次成功时直接用那一次的结果。
    bad_citations 会从倒序空间映射回原始编号，保证两个方向可比。
    """
    n = len(sources)
    forward = await judge.judge(question, sources, answer)
    reverse = await judge.judge(question, reorder_sources(sources, reverse=True), answer)

    if forward is None and reverse is None:
        return None
    if forward is None:
        assert reverse is not None
        mapped = [n - j + 1 for j in reverse.bad_citations if 1 <= j <= n]
        return RobustVerdict(
            grounded=float(reverse.grounded),
            relevant=float(reverse.relevant),
            bad_citations=sorted(set(mapped)),
            order_delta=0,
            forward=reverse,
            reverse=reverse,
        )
    if reverse is None:
        return RobustVerdict(
            grounded=float(forward.grounded),
            relevant=float(forward.relevant),
            bad_citations=list(forward.bad_citations),
            order_delta=0,
            forward=forward,
            reverse=forward,
        )

    mapped_reverse = [n - j + 1 for j in reverse.bad_citations if 1 <= j <= n]
    return RobustVerdict(
        grounded=(forward.grounded + reverse.grounded) / 2,
        relevant=(forward.relevant + reverse.relevant) / 2,
        bad_citations=sorted(set(forward.bad_citations) | set(mapped_reverse)),
        order_delta=abs(forward.grounded - reverse.grounded),
        forward=forward,
        reverse=reverse,
    )
