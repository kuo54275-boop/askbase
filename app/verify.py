"""引用校验（规则版，完全不依赖模型）。

为什么需要它：LLM 裁判可能和生成模型有相同的偏好，规则校验不会。
它只回答一个很硬的问题——**答案里那句话引用的那份资料，真的支持这句话吗？**

两条判据：
1. 数字一致性：结论里的每个数字，必须出现在它引用的资料里。
   数字是最容易编造、也最容易被抓的证据（"15 分钟" vs "4 小时"）。
2. 术语覆盖：句子里的中文二元组，有多少能在引用资料中找到。
   没有数字的句子（如"由直属主管审批"）只能靠这条兜。

规则版会有误判（改写、同义替换、跨句引用），所以它的定位是**下限估计**：
规则通过率低，说明引用大概率有问题；规则通过率高，不能证明一定没问题。
"""

from __future__ import annotations

import re

from .models import Source

_NUM_RE = re.compile(r"\d+(?:\.\d+)?")
_CITE_RE = re.compile(r"\[(\d{1,2})\]")
_SENT_SPLIT = re.compile(r"(?<=[。！？!?])")

# 高频虚词组成的二元组不具备区分度，参与匹配只会抬高覆盖率
_STOP_BIGRAMS = {
    "结论", "依据", "资料", "根据", "可以", "需要", "应当", "如果", "以及",
    "或者", "但是", "因此", "所以", "这个", "那个", "我们", "他们", "以下",
}

_TERM_COVERAGE_THRESHOLD = 0.5

# 「依据：…上述…」这类总结句不是事实断言，而是对前文的指代，不参与引用校验。
# 实测两次误伤都出在这种句子上（术语覆盖率天然低，因为它本来就没带多少原文内容）。
_META_RE = re.compile(r"^\s*(依据|参考|详见|参见|来源)|上述|前者|该规定|综上")

# 否定断言：「资料中未提及 X」。它在字面上永远找不到"支持"，
# 但这恰恰是模型最该做的事（拒答），不能算引用错误。
_NEGATIVE_RE = re.compile(r"未提及|没有提到|未提到|未包含|未说明|没有说明|无相关|不涉及|未规定")


def is_meta_claim(claim: str) -> bool:
    """判断是不是"指代式总结句"（依据：…上述…）。"""
    return bool(_META_RE.search(claim))


def is_negative_claim(claim: str) -> bool:
    """判断是不是"否定断言"（资料中未提及 X）。"""
    return bool(_NEGATIVE_RE.search(claim))


def is_unverifiable_claim(claim: str) -> bool:
    """这两类句子都不该参与字面校验。"""
    return is_meta_claim(claim) or is_negative_claim(claim)


def _norm(text: str) -> str:
    return re.sub(r"\s+", "", text)


def split_claims(answer: str) -> list[str]:
    """把答案切成句子级的"结论"，用于逐条校验。

    注意角标的位置：真实答案长这样「…满5年15天。[1]」，句号在角标前面。
    按句号直接切会把 [1] 切到下一条结论上，所以这里把开头的角标回挂到上一句。
    """
    parts = [p for p in _SENT_SPLIT.split(answer) if p and p.strip()]
    claims: list[str] = []

    for part in parts:
        match = re.match(r"\s*((?:\[\d{1,2}\]\s*)+)", part)
        if match and claims:
            claims[-1] += match.group(1).strip()
            part = part[match.end():]
        if part.strip():
            claims.append(part.strip())
    return claims


def _terms(text: str) -> list[str]:
    """抽取中文二元组作为可匹配的特征。"""
    text = _norm(_CITE_RE.sub("", text))
    terms: list[str] = []
    for run in re.findall(r"[\u4e00-\u9fff]+", text):
        for i in range(len(run) - 1):
            bigram = run[i : i + 2]
            if bigram not in _STOP_BIGRAMS:
                terms.append(bigram)
    return terms


def check_claim(claim: str, source_texts: list[str]) -> tuple[bool, str]:
    """判断一条结论是否被它引用的资料支持。

    source_texts 是这句话**引用到的全部**资料，不是单份：
    「文档保留 90 天 [1]；打印内容 24 小时 [2]」这种一句话带两个来源的情况，
    如果拿 90 和 24 分别去对单个来源，必然误伤。所以：
    - 数字：必须出现在引用资料的并集里；
    - 术语：取这几份资料里覆盖率最高的那个。
    """
    numbers = _NUM_RE.findall(_CITE_RE.sub("", claim))
    normalized = [_norm(t) for t in source_texts]

    if numbers:
        union = "".join(normalized)
        missing = [n for n in numbers if n not in union]
        if missing:
            return False, f"数字 {missing} 未出现在引用资料中"
        return True, "数字一致"

    terms = _terms(claim)
    if not terms:
        return True, "无可校验内容"

    best = max(
        (sum(1 for t in terms if t in src) / len(terms) for src in normalized),
        default=0.0,
    )
    return best >= _TERM_COVERAGE_THRESHOLD, f"术语覆盖 {best:.0%}"


def verify_answer(answer: str, sources: list[Source]) -> dict:
    """校验答案里的引用角标，返回统计和问题列表。

    两级判定：
    1. 先按「角标 -> 单份资料」严格校验（能抓张冠李戴）；
    2. 严格校验失败且这句话引用了多份资料时，再用并集校验一次
       （避免「90 天 [1]；24 小时 [2]」这种一句话跨来源的误伤）。
    """
    claims = split_claims(answer)
    checked = 0
    passed = 0
    skipped = 0
    issues: list[dict] = []

    for claim in claims:
        if is_unverifiable_claim(claim):
            skipped += 1
            continue
        cites = [int(c) for c in _CITE_RE.findall(claim)]
        if not cites:
            continue

        for n in cites:
            if not (1 <= n <= len(sources)):
                checked += 1
                issues.append(
                    {"claim": claim[:80], "cite": n, "reason": "引用了不存在的来源编号"}
                )

        valid = [(n, sources[n - 1].text) for n in cites if 1 <= n <= len(sources)]
        if not valid:
            continue

        checked += len(valid)
        strict_failed: list[tuple[int, str]] = []
        for n, text in valid:
            ok, reason = check_claim(claim, [text])
            if not ok:
                strict_failed.append((n, reason))
        if not strict_failed:
            passed += len(valid)
            continue

        union_ok, union_reason = check_claim(claim, [t for _, t in valid])
        if union_ok and len(valid) > 1:
            passed += len(valid)
            continue

        for n, reason in strict_failed:
            issues.append(
                {
                    "claim": claim[:80],
                    "cite": n,
                    "reason": reason if not union_ok else union_reason,
                }
            )

    return {
        "claims": len(claims),
        "checked": checked,
        "passed": passed,
        "skipped": skipped,
        "rate": (passed / checked) if checked else None,
        "issues": issues,
    }
