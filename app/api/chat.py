"""问答接口：SSE 流式返回。

为什么用 SSE 而不是 WebSocket：
问答是「客户端发一次、服务端推多次」的单向流，SSE 基于普通 HTTP，
天然兼容反向代理、断线重连、HTTP/2，实现和运维成本都比 WebSocket 低。
"""

from __future__ import annotations

import json
from typing import AsyncIterator

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from ..models import ChatRequest

router = APIRouter(tags=["chat"])

_SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",  # 关闭 Nginx 缓冲，否则流式会被攒成一次性输出
}


def _sse(event: str, payload: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


@router.post("/chat", summary="流式问答（SSE）")
async def chat(request: Request, body: ChatRequest) -> StreamingResponse:
    service = request.app.state.service

    async def event_gen() -> AsyncIterator[str]:
        try:
            async for event in service.astream(
                body.question, body.top_k, body.history
            ):
                kind = event["type"]
                if kind == "rewrite":
                    yield _sse("rewrite", {"question": event["question"]})
                elif kind == "sources":
                    yield _sse("sources", {"sources": event["sources"]})
                elif kind == "delta":
                    yield _sse("delta", {"text": event["text"]})
                elif kind == "error":
                    yield _sse("error", {"message": event["message"]})
                elif kind == "done":
                    yield _sse("done", event)
        except Exception as exc:  # 兜底：保证前端一定能收到终止事件
            yield _sse("error", {"message": f"服务异常：{exc}"})

    return StreamingResponse(event_gen(), media_type="text/event-stream", headers=_SSE_HEADERS)
