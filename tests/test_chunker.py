"""切分逻辑单测：标题识别、长度约束、overlap 行为。"""

from __future__ import annotations

from app.ingest.chunker import DEFAULT_SECTION, chunk_text, index_text, is_heading


def test_is_heading():
    assert is_heading("## 报销流程")
    assert is_heading("第三章 差旅标准")
    assert is_heading("一、考勤制度")
    assert not is_heading("这是一段普通的正文，不是标题。")
    assert not is_heading("很长" * 30)


def test_sections_are_detected():
    text = "## 报销流程\n先提交申请。\n\n## 差旅标准\n住宿每晚不超过 600 元。"
    pieces = chunk_text(text, size=200, overlap=0)
    sections = [p.section for p in pieces]
    assert "报销流程" in sections
    assert "差旅标准" in sections


def test_no_heading_falls_back_to_default_section():
    pieces = chunk_text("这里没有任何标题，只有一段正文。", size=200, overlap=0)
    assert len(pieces) == 1
    assert pieces[0].section == DEFAULT_SECTION


def test_chunk_size_is_bounded():
    text = "这是一句测试用的中文句子，用来验证切分长度是否受控。" * 100
    size = 200
    pieces = chunk_text(text, size=size, overlap=40)
    assert len(pieces) > 1
    # 允许因为句子边界而略超，但不能翻倍
    assert all(len(p.text) <= size * 1.3 for p in pieces)


def test_overlap_keeps_context():
    sentences = [f"第{i}句话用来填充长度。" for i in range(60)]
    text = "".join(sentences)
    pieces = chunk_text(text, size=150, overlap=60)
    assert len(pieces) >= 2
    # 相邻切片应当有公共内容（overlap 生效）
    overlap_found = False
    for prev, nxt in zip(pieces, pieces[1:]):
        tail = prev.text[-20:]
        if tail and tail in nxt.text:
            overlap_found = True
            break
    assert overlap_found


def test_ordinals_are_continuous():
    text = "## A\n" + "内容。" * 200 + "\n## B\n" + "内容。" * 200
    pieces = chunk_text(text, size=100, overlap=20)
    assert [p.ordinal for p in pieces] == list(range(len(pieces)))


def test_index_text_prefixes_section():
    assert index_text("报销流程", "正文内容").startswith("报销流程")
    assert index_text(DEFAULT_SECTION, "正文内容") == "正文内容"
