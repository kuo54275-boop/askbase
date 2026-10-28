FROM python:3.12-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    LLM_PROVIDER=auto

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app ./app
COPY eval ./eval
COPY scripts ./scripts
COPY data/samples ./data/samples

EXPOSE 8000

# 首次启动自动导入示例文档（知识库为空时）
CMD ["sh", "-c", "python scripts/seed.py && uvicorn app.main:app --host 0.0.0.0 --port 8000"]
