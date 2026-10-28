"""接口层集成测试：上传 -> 列表 -> 流式问答 -> 删除 -> 评测。

全程使用 Mock LLM，不消耗 token、不依赖网络，可以直接进 CI。
"""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app

TEST_ROOT = Path(__file__).resolve().parent / ".tmp"

DOC = """
## 报销流程

所有费用报销统一在 OA 系统提交。报销需在费用发生后 30 天内提交。

必须上传的材料包括：发票原件照片、消费明细小票、出差审批单。

## 年假制度

员工入职满 1 年享有 5 天带薪年假，当年未休完最多可结转 5 天。
""".encode("utf-8")


@pytest.fixture()
def client():
    """每个用例独立的数据目录，用例之间互不影响。"""
    data_dir = TEST_ROOT / uuid.uuid4().hex[:8]
    settings = Settings(
        _env_file=None,
        data_dir=data_dir,
        llm_provider="mock",
        deepseek_api_key="",
        top_k=3,
        chunk_size=120,
        chunk_overlap=20,
    )
    settings.ensure_dirs()
    app = create_app(settings)
    with TestClient(app) as c:
        yield c
    shutil.rmtree(data_dir, ignore_errors=True)


def _upload(client, name="员工手册.txt", data=DOC):
    res = client.post(
        "/api/docs/upload",
        files={"file": (name, data, "text/plain")},
    )
    assert res.status_code == 200, res.text
    return res.json()


def test_healthz(client):
    res = client.get("/healthz")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ok"
    assert body["llm_provider"] == "mock"
    assert body["n_docs"] == 0


def test_upload_and_list(client):
    out = _upload(client)
    assert out["doc"]["n_chunks"] >= 1
    assert "报销" in out["preview"]

    docs = client.get("/api/docs").json()
    assert len(docs) == 1
    assert docs[0]["name"] == "员工手册.txt"

    health = client.get("/healthz").json()
    assert health["n_docs"] == 1
    assert health["n_chunks"] == out["doc"]["n_chunks"]


def test_reupload_same_name_replaces(client):
    first = _upload(client)
    assert first["replaced"] is False
    second = _upload(client)
    # 同名文件覆盖，而不是变成两份
    assert second["replaced"] is True
    docs = client.get("/api/docs").json()
    assert len(docs) == 1
    assert docs[0]["id"] == second["doc"]["id"]
    assert docs[0]["id"] != first["doc"]["id"]


def test_upload_rejects_unsupported_type(client):
    res = client.post(
        "/api/docs/upload",
        files={"file": ("a.exe", b"binary", "application/octet-stream")},
    )
    assert res.status_code == 400


def test_chat_streams_answer_with_sources(client):
    _upload(client)
    with client.stream(
        "POST", "/api/chat", json={"question": "报销需要哪些材料？"}
    ) as res:
        assert res.status_code == 200
        assert res.headers["content-type"].startswith("text/event-stream")
        body = "".join(res.iter_text())

    assert "event: sources" in body
    assert "event: delta" in body
    assert "event: done" in body
    # 命中内容应包含关键实体
    assert "发票" in body


def test_chat_without_docs_gives_hint(client):
    with client.stream("POST", "/api/chat", json={"question": "随便问问"}) as res:
        body = "".join(res.iter_text())
    assert "event: done" in body
    assert "上传" in body


def test_chat_with_history_triggers_rewrite(client):
    _upload(client)
    body = {
        "question": "那需要几天？",
        "history": [
            {"role": "user", "content": "年假有几天？"},
            {"role": "assistant", "content": "满一年 5 天 [1]"},
        ],
    }
    with client.stream("POST", "/api/chat", json=body) as res:
        assert res.status_code == 200
        text = "".join(res.iter_text())

    # 有历史时必须触发改写事件，且改写结果要包含上一轮的关键实体
    assert "event: rewrite" in text
    assert "年假" in text
    assert "event: done" in text


def test_chat_rejects_bad_history_role(client):
    res = client.post(
        "/api/chat",
        json={"question": "q", "history": [{"role": "system", "content": "x"}]},
    )
    assert res.status_code == 422


def test_delete_document(client):
    out = _upload(client)
    doc_id = out["doc"]["id"]
    assert client.delete(f"/api/docs/{doc_id}").status_code == 200
    assert client.get("/api/docs").json() == []
    # 重复删除返回 404
    assert client.delete(f"/api/docs/{doc_id}").status_code == 404


def test_eval_endpoint_runs(client):
    _upload(client, name="员工手册.txt", data=DOC)
    res = client.post("/api/eval/run")
    # 评测集里要求的是"员工手册.md"，这里用 .txt 会全部未命中，但接口本身必须正常
    assert res.status_code == 200
    body = res.json()
    assert body["total"] > 0
    assert "@1" in body["recall_at"]
