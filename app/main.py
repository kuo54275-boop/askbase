"""应用装配：依赖注入 + 生命周期。

检索索引在启动时从 SQLite 重建，进程重启不丢数据，也不需要单独的索引服务。
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .api import chat, docs, eval_api
from .config import BASE_DIR, Settings, get_settings
from .db import Repository
from .llm import build_llm
from .models import HealthInfo
from .rag import RagService

STATIC_DIR = BASE_DIR / "app" / "static"


def create_app(settings: Settings | None = None) -> FastAPI:
    """工厂函数：测试时可以注入独立的数据目录，避免污染真实数据。"""
    settings = settings or get_settings()
    settings.ensure_dirs()

    repo = Repository(settings.db_path)
    repo.init_db()

    llm = build_llm(
        provider=settings.effective_llm_provider,
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
        model=settings.chat_model,
        timeout=settings.llm_timeout,
        max_retries=settings.llm_max_retries,
    )
    service = RagService(settings, repo, llm)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        service.refresh_index()
        yield

    app = FastAPI(
        title="AskBase · 知识库问答",
        description="基于 RAG 的企业知识库问答系统：混合检索 + 引用溯源 + 流式输出",
        version="1.0.0",
        lifespan=lifespan,
    )

    # 统一把依赖挂到 app.state，路由里通过 Depends 取
    app.state.settings = settings
    app.state.repo = repo
    app.state.service = service

    app.include_router(docs.router, prefix="/api")
    app.include_router(chat.router, prefix="/api")
    app.include_router(eval_api.router, prefix="/api")

    @app.get("/healthz", response_model=HealthInfo, tags=["system"])
    def healthz() -> HealthInfo:
        n_docs, n_chunks = repo.stats()
        return HealthInfo(
            status="ok",
            llm_provider=llm.name,
            embedding_provider=settings.embedding_provider,
            n_docs=n_docs,
            n_chunks=n_chunks,
        )

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    return app


app = create_app()
