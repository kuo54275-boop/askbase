"""轻量分词器。

为什么不用 jieba：
- 少一个依赖，离线环境零安装；
- 中文检索场景下「单字 + 相邻二字」的组合对 BM25 已经够用，
  而且天然支持任意新词，不需要维护词典。

生产环境可以直接换成 jieba / HanLP，接口不变。
"""

from __future__ import annotations

import re

_ASCII_WORD = re.compile(r"[a-z0-9][a-z0-9_\-#+.]*")
_CJK_RUN = re.compile(r"[\u4e00-\u9fff]+")


def tokenize(text: str) -> list[str]:
    """英文按词、中文按「单字 + bigram」切分。"""
    text = text.lower()
    tokens: list[str] = _ASCII_WORD.findall(text)
    for run in _CJK_RUN.findall(text):
        tokens.extend(run)  # unigram：保证召回
        tokens.extend(run[i : i + 2] for i in range(len(run) - 1))  # bigram：保证精度
    return tokens
