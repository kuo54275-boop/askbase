"""全局配置。

所有可调参数集中在这里，通过 .env 覆盖，方便面试时讲"配置与代码分离"。
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# 项目根目录（app/ 的上一级）
BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---------- 大模型 ----------
    # 方式 A（通用）：任何 OpenAI 兼容端点 —— OpenAI / 通义 / Kimi / 智谱 / 硅基流动 / Ollama / vLLM
    # 只要填了 LLM_API_KEY + LLM_BASE_URL，就优先用这套，DEEPSEEK_* 会被忽略
    llm_api_key: str = ""
    llm_base_url: str = ""
    llm_model: str = ""

    # 方式 B（DeepSeek 专用，历史配置，仍然兼容）
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    chat_model: str = "deepseek-chat"

    # auto: 有 key 用真实模型，没 key 自动降级 Mock（保证任何环境都能跑起来）
    llm_provider: str = "auto"
    llm_timeout: float = 60.0
    llm_max_retries: int = 2
    temperature: float = 0.2
    max_tokens: int = 900

    # ---------- 检索 ----------
    # hashing: 零依赖的字符 n-gram 哈希向量；后续可换 bge / 外部 embedding API
    embedding_provider: str = "hashing"
    top_k: int = 5
    rrf_k: int = 60
    # RRF 融合权重。默认不是 1:1 —— 消融显示哈希向量是弱信号，
    # 等权融合会把它的噪声放大到能盖过 BM25 的首位（Recall@1 92.45% vs 94.34%）。
    # 换成 bge 这类稠密向量后，应把 vector_weight 调回 0.6~1.0 重新评测。
    bm25_weight: float = 0.6
    vector_weight: float = 0.2
    # 门控：低于 top * ratio 的候选直接丢弃。0.65 是在 53 条评测集上
    # "召回不变、平均来源数 2.21 -> 1.94" 的结果，门控越紧 prompt 噪声越少。
    bm25_min_ratio: float = 0.65
    min_cos_ratio: float = 0.4
    # 重排：none | llm。llm 用模型给候选打分（不下载模型、不占磁盘），
    # 代价是每个 query 多一次 API 调用；Mock 模式会自动降级为 none。
    rerank_provider: str = "llm"
    rerank_candidates: int = 10

    # ---------- 跨厂商裁判（只用于评测复核，不参与线上问答） ----------
    # 三项都填了才会启用；否则回退到主模型当裁判。
    judge_api_key: str = ""
    judge_base_url: str = ""
    judge_model: str = ""

    # ---------- 切分 ----------
    chunk_size: int = 400
    chunk_overlap: int = 60

    # ---------- 存储 ----------
    data_dir: Path = BASE_DIR / "data"

    @property
    def docs_dir(self) -> Path:
        return self.data_dir / "docs"

    @property
    def samples_dir(self) -> Path:
        return self.data_dir / "samples"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "app.db"

    @property
    def effective_llm_provider(self) -> str:
        """把 auto 解析成真实 provider。

        优先级：显式指定 > 通用 OpenAI 兼容配置 > DeepSeek 专用配置 > Mock。
        """
        if self.llm_provider != "auto":
            return self.llm_provider
        if self.llm_api_key.strip() and self.llm_base_url.strip():
            return "openai"
        if self.deepseek_api_key.strip():
            return "deepseek"
        return "mock"

    # 下面三个属性把两套配置收敛成一份，调用方不用关心用户填的是哪种
    @property
    def active_api_key(self) -> str:
        return self.llm_api_key.strip() or self.deepseek_api_key.strip()

    @property
    def active_base_url(self) -> str:
        return (self.llm_base_url.strip() or self.deepseek_base_url).rstrip("/")

    @property
    def active_model(self) -> str:
        return self.llm_model.strip() or self.chat_model

    @property
    def is_configured(self) -> bool:
        """是否已接入真实模型（未接入时跑 Mock）。"""
        return self.effective_llm_provider != "mock"

    def ensure_dirs(self) -> None:
        for p in (self.data_dir, self.docs_dir, self.samples_dir):
            p.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_dirs()
    return settings
