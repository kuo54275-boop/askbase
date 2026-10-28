"""SQLite 持久层。

选 SQLite 而不是 Postgres + 向量库，是刻意的工程取舍：
秋招项目要的是「克隆下来就能跑」，SQLite 零运维、单文件、可随代码提交。
切片向量不落库，启动时从文本重算（哈希向量是确定性的），
换 embedding 模型时也不会出现「库里的向量和代码不一致」这种脏数据问题。
"""

from __future__ import annotations

import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from .models import DocMeta

_SCHEMA = """
CREATE TABLE IF NOT EXISTS docs (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    ext         TEXT NOT NULL,
    size        INTEGER NOT NULL,
    n_chunks    INTEGER NOT NULL,
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
    id        TEXT PRIMARY KEY,
    doc_id    TEXT NOT NULL REFERENCES docs(id) ON DELETE CASCADE,
    ordinal   INTEGER NOT NULL,
    section   TEXT NOT NULL DEFAULT '',
    text      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);

CREATE TABLE IF NOT EXISTS qa_logs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    question    TEXT NOT NULL,
    answer      TEXT NOT NULL,
    n_sources   INTEGER NOT NULL,
    latency_ms  INTEGER NOT NULL,
    provider    TEXT NOT NULL,
    created_at  TEXT NOT NULL
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


class Repository:
    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def init_db(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as conn:
            conn.executescript(_SCHEMA)

    # ---------- 文档 ----------
    def add_document(
        self,
        name: str,
        ext: str,
        size: int,
        chunks: list[tuple[int, str, str]],
    ) -> DocMeta:
        """chunks: [(ordinal, section, text)]"""
        doc_id = uuid.uuid4().hex[:12]
        created = _now()
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO docs (id, name, ext, size, n_chunks, created_at) VALUES (?,?,?,?,?,?)",
                (doc_id, name, ext, size, len(chunks), created),
            )
            conn.executemany(
                "INSERT INTO chunks (id, doc_id, ordinal, section, text) VALUES (?,?,?,?,?)",
                [
                    (f"{doc_id}-{ordinal}", doc_id, ordinal, section, text)
                    for ordinal, section, text in chunks
                ],
            )
        return DocMeta(
            id=doc_id, name=name, ext=ext, size=size, n_chunks=len(chunks), created_at=created
        )

    def list_documents(self) -> list[DocMeta]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT id, name, ext, size, n_chunks, created_at FROM docs ORDER BY created_at DESC"
            ).fetchall()
        return [DocMeta(**dict(r)) for r in rows]

    def find_by_name(self, name: str) -> DocMeta | None:
        """同名文档只保留最新一份：重复上传视为更新，而不是新增。"""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT id, name, ext, size, n_chunks, created_at FROM docs"
                " WHERE name = ? ORDER BY created_at DESC LIMIT 1",
                (name,),
            ).fetchone()
        return DocMeta(**dict(row)) if row else None

    def delete_document(self, doc_id: str) -> bool:
        with self._conn() as conn:
            cur = conn.execute("DELETE FROM docs WHERE id = ?", (doc_id,))
            conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
            return cur.rowcount > 0

    # ---------- 切片 ----------
    def iter_chunks(self) -> list[dict]:
        """返回全部切片（含所属文档名），用于重建检索索引。"""
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT c.id, c.doc_id, c.ordinal, c.section, c.text, d.name AS doc_name
                FROM chunks c JOIN docs d ON d.id = c.doc_id
                ORDER BY c.doc_id, c.ordinal
                """
            ).fetchall()
        return [dict(r) for r in rows]

    # ---------- 问答日志 ----------
    def log_qa(self, question: str, answer: str, n_sources: int, latency_ms: int, provider: str) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO qa_logs (question, answer, n_sources, latency_ms, provider, created_at)"
                " VALUES (?,?,?,?,?,?)",
                (question, answer, n_sources, latency_ms, provider, _now()),
            )

    # ---------- 统计 ----------
    def stats(self) -> tuple[int, int]:
        with self._conn() as conn:
            n_docs = conn.execute("SELECT COUNT(*) FROM docs").fetchone()[0]
            n_chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        return n_docs, n_chunks
