"""中文友好的文档切分。

设计要点（面试常问）：
1. 先按"标题"切成 section，保留章节路径 —— 检索命中后可以告诉用户"这段来自哪一章"。
2. section 内按中文句子边界切，再做"句子级滑动窗口"，避免把一句话从中间切断。
3. overlap 保证跨 chunk 的语义不被切断，代价是少量冗余（评测里可量化）。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

DEFAULT_SECTION = "正文"

# 标题识别：Markdown 标题 / "第X章|节|部分" / 中文序号 / "1.2 小节"
_HEADING_PATTERNS = [
    re.compile(r"^#{1,6}\s+\S"),
    re.compile(r"^第[一二三四五六七八九十百零\d]+[章节部分条]\s*"),
    re.compile(r"^[一二三四五六七八九十]+、\s*\S"),
    re.compile(r"^\d+(\.\d+){0,2}[、.]?\s+\S"),
]

# 中文句末标点 + 英文句末标点 + 换行
_SENT_SPLIT = re.compile(r"(?<=[。！？；!?;])|(?<=\n)")

_MAX_HEADING_LEN = 40


def is_heading(line: str) -> bool:
    """短行 + 符合标题特征 => 当成标题。"""
    s = line.strip()
    if not s or len(s) > _MAX_HEADING_LEN:
        return False
    # 以句末标点结尾的通常是正文句子，不是标题
    if s[-1] in "。！？；，,.;":
        return False
    return any(p.match(s) for p in _HEADING_PATTERNS)


@dataclass
class ChunkPiece:
    section: str
    text: str
    ordinal: int


def split_sections(text: str) -> list[tuple[str, str]]:
    """把正文按标题切成 (section, body) 列表。"""
    sections: list[tuple[str, str]] = []
    current = DEFAULT_SECTION
    buffer: list[str] = []

    for raw_line in text.split("\n"):
        line = raw_line.strip()
        if not line:
            continue
        if is_heading(line):
            if buffer:
                sections.append((current, "\n".join(buffer)))
                buffer = []
            current = line.lstrip("#").strip() or DEFAULT_SECTION
        else:
            buffer.append(line)
    if buffer:
        sections.append((current, "\n".join(buffer)))

    # 整篇没有识别出标题时，退化成单 section
    return sections or [(DEFAULT_SECTION, text)]


def _split_sentences(body: str) -> list[str]:
    parts = [p for p in _SENT_SPLIT.split(body) if p and p.strip()]
    return parts


def _hard_split(sentence: str, size: int) -> list[str]:
    """超长句子（如一个长列表项）按 size 硬切，避免单片过大。"""
    return [sentence[i : i + size] for i in range(0, len(sentence), size)]


def _pack(sentences: list[str], size: int, overlap: int) -> list[str]:
    """句子级滑动窗口打包。"""
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for sent in sentences:
        pieces = _hard_split(sent, size) if len(sent) > size else [sent]
        for piece in pieces:
            if current and current_len + len(piece) > size:
                chunks.append("".join(current).strip())
                # 用尾部句子构造 overlap，而不是按字符硬截断
                tail: list[str] = []
                tail_len = 0
                for prev in reversed(current):
                    if tail_len + len(prev) > overlap:
                        break
                    tail.insert(0, prev)
                    tail_len += len(prev)
                current, current_len = tail, tail_len
            current.append(piece)
            current_len += len(piece)

    if current:
        chunks.append("".join(current).strip())
    return [c for c in chunks if c]


def chunk_text(
    text: str,
    size: int = 400,
    overlap: int = 60,
) -> list[ChunkPiece]:
    """主入口：文本 -> 带章节信息的切片列表。"""
    if size <= 0:
        raise ValueError("size 必须大于 0")
    overlap = max(0, min(overlap, size // 2))

    pieces: list[ChunkPiece] = []
    ordinal = 0
    for section, body in split_sections(text):
        for chunk in _pack(_split_sentences(body), size, overlap):
            pieces.append(ChunkPiece(section=section, text=chunk, ordinal=ordinal))
            ordinal += 1
    return pieces


def index_text(section: str, text: str) -> str:
    """用于建索引的文本：把章节名拼进去，短切片也能命中主题词。"""
    if section and section != DEFAULT_SECTION:
        return f"{section}\n{text}"
    return text
