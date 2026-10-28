"""文档解析：把上传的文件统一转成纯文本。

支持 .txt / .md / .pdf / .docx，按扩展名分发。
解析失败统一抛 ParsingError，由 API 层转成 400，避免 500 泄漏堆栈。
"""

from __future__ import annotations

import io
from pathlib import Path

SUPPORTED_EXTS = {".txt", ".md", ".markdown", ".pdf", ".docx"}


class ParsingError(Exception):
    """文档无法解析成文本。"""


def _decode_text(data: bytes) -> str:
    """中文场景下编码很杂，按 utf-8 -> utf-8-sig -> gbk 依次尝试。"""
    for enc in ("utf-8", "utf-8-sig", "gb18030", "gbk"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="ignore")


def _from_pdf(data: bytes) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - 依赖缺失时给出明确提示
        raise ParsingError("缺少 pypdf 依赖，无法解析 PDF") from exc

    try:
        reader = PdfReader(io.BytesIO(data))
        pages = [(page.extract_text() or "") for page in reader.pages]
    except Exception as exc:  # pypdf 对损坏文件抛的异常类型不统一
        raise ParsingError(f"PDF 解析失败：{exc}") from exc
    return "\n\n".join(p.strip() for p in pages if p.strip())


def _from_docx(data: bytes) -> str:
    try:
        import docx
    except ImportError as exc:  # pragma: no cover
        raise ParsingError("缺少 python-docx 依赖，无法解析 DOCX") from exc

    try:
        document = docx.Document(io.BytesIO(data))
    except Exception as exc:
        raise ParsingError(f"DOCX 解析失败：{exc}") from exc

    lines: list[str] = []
    for para in document.paragraphs:
        text = para.text.strip()
        if text:
            lines.append(text)
    # 表格内容也要能检索到
    for table in document.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells if c.text.strip()]
            if cells:
                lines.append(" | ".join(cells))
    return "\n".join(lines)


def extract_text(filename: str, data: bytes) -> str:
    """按扩展名解析文档，返回规整后的纯文本。"""
    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED_EXTS:
        raise ParsingError(f"不支持的文件类型 {ext or '(无扩展名)'}，仅支持 {sorted(SUPPORTED_EXTS)}")

    if not data:
        raise ParsingError("文件内容为空")

    if ext == ".pdf":
        text = _from_pdf(data)
    elif ext == ".docx":
        text = _from_docx(data)
    else:
        text = _decode_text(data)

    text = normalize_text(text)
    if not text.strip():
        raise ParsingError("未能从文档中提取到文本（可能是扫描版 PDF，需要 OCR）")
    return text


def normalize_text(text: str) -> str:
    """统一换行、去掉多余空白，让切分结果更稳定。"""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in text.split("\n")]
    text = "\n".join(lines)
    # 三个以上连续空行压成一个
    while "\n\n\n" in text:
        text = text.replace("\n\n\n", "\n\n")
    return text.strip()
