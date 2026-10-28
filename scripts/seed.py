"""把 data/samples 下的示例文档导入知识库。

用法：
    python -m scripts.seed          # 增量导入（同名跳过）
    python -m scripts.seed --reset  # 清空后重新导入
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings  # noqa: E402
from app.db import Repository  # noqa: E402
from app.llm import build_llm  # noqa: E402
from app.rag import RagService  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="先清空已有文档")
    args = parser.parse_args()

    settings = get_settings()
    repo = Repository(settings.db_path)
    repo.init_db()

    if args.reset:
        for doc in repo.list_documents():
            repo.delete_document(doc.id)
        print("已清空知识库")

    llm = build_llm(
        settings.effective_llm_provider,
        settings.deepseek_api_key,
        settings.deepseek_base_url,
        settings.chat_model,
        settings.llm_timeout,
        settings.llm_max_retries,
    )
    service = RagService(settings, repo, llm)

    existing = {d.name for d in repo.list_documents()}
    files = sorted(p for p in settings.samples_dir.iterdir() if p.is_file())
    if not files:
        print(f"示例目录为空：{settings.samples_dir}")
        return 1

    for path in files:
        if path.name in existing and not args.reset:
            print(f"跳过（已存在）{path.name}")
            continue
        doc, _, _ = service.ingest(path.name, path.read_bytes())
        print(f"导入 {path.name} -> {doc['n_chunks']} 个切片")

    n_docs, n_chunks = repo.stats()
    print(f"完成：{n_docs} 篇文档 / {n_chunks} 个切片")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
