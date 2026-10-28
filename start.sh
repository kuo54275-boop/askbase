#!/usr/bin/env bash
# AskBase 一键启动（macOS / Linux）
# 用法：./start.sh [端口]
set -e
cd "$(dirname "$0")"

PORT="${1:-8000}"

echo "============================================"
echo "  AskBase - 企业知识库问答"
echo "============================================"
echo

if ! command -v python3 >/dev/null 2>&1; then
    echo "[X] 没有找到 python3，请先安装 Python 3.11+"
    exit 1
fi

if [ ! -x ".venv/bin/python" ]; then
    echo "[1/3] 首次运行，创建虚拟环境..."
    python3 -m venv .venv
    echo "[2/3] 安装依赖，第一次需要几分钟..."
    .venv/bin/python -m pip install -r requirements.txt
else
    echo "[1/3] 虚拟环境已就绪"
fi

if [ ! -f "data/app.db" ]; then
    echo "[2/3] 导入示例文档..."
    .venv/bin/python scripts/seed.py
else
    echo "[2/3] 知识库已就绪"
fi

if [ ! -f ".env" ]; then
    echo
    echo "[!] 没有找到 .env，将以 Mock 模式启动（答案由检索结果抽取生成）"
    echo "    接入真实模型：cp .env.example .env，填入 LLM_API_KEY 后重新启动"
    echo
fi

echo "[3/3] 启动服务: http://127.0.0.1:${PORT}"
echo "      浏览器会自动打开；按 Ctrl+C 停止服务"
echo

# 后台延迟打开浏览器，避免服务还没起来就访问
(
    sleep 4
    if command -v xdg-open >/dev/null 2>&1; then
        xdg-open "http://127.0.0.1:${PORT}" >/dev/null 2>&1 || true
    elif command -v open >/dev/null 2>&1; then
        open "http://127.0.0.1:${PORT}" >/dev/null 2>&1 || true
    fi
) &

exec .venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port "${PORT}"
