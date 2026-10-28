"""端到端自检：不依赖浏览器，验证「服务起来 -> 检索 -> 流式输出」整条链路。

用法：
    python scripts/smoke.py                 # 默认 http://127.0.0.1:8000
    python scripts/smoke.py --base http://127.0.0.1:8123
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import httpx


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:8000")
    parser.add_argument("--question", default="报销需要提交哪些材料？")
    args = parser.parse_args()
    base = args.base.rstrip("/")

    # 1. 等服务就绪
    health = None
    for _ in range(40):
        try:
            health = httpx.get(f"{base}/healthz", timeout=2).json()
            break
        except Exception:
            time.sleep(0.5)
    if not health:
        print("服务未就绪")
        return 1
    print(f"healthz   : {health}")

    # 2. 文档列表
    docs = httpx.get(f"{base}/api/docs", timeout=10).json()
    print(f"文档数    : {len(docs)} -> {[d['name'] for d in docs]}")
    if not docs:
        print("知识库为空，先运行 python scripts/seed.py")
        return 1

    # 3. 首页
    index = httpx.get(f"{base}/", timeout=10)
    print(f"首页      : {index.status_code}, {len(index.text)} bytes")

    # 4. 流式问答
    started = time.perf_counter()
    counts: dict[str, int] = {}
    answer = ""
    sources: list[dict] = []
    done: dict = {}

    with httpx.stream(
        "POST", f"{base}/api/chat", json={"question": args.question}, timeout=120
    ) as resp:
        if resp.status_code != 200:
            print(f"问答接口异常: {resp.status_code}")
            return 1
        event = None
        for line in resp.iter_lines():
            if line.startswith("event:"):
                event = line[6:].strip()
            elif line.startswith("data:"):
                payload = json.loads(line[5:].strip())
                counts[event] = counts.get(event, 0) + 1
                if event == "sources":
                    sources = payload["sources"]
                elif event == "delta":
                    answer += payload["text"]
                elif event == "done":
                    done = payload

    print(f"事件统计  : {counts}")
    for s in sources:
        print(f"  来源 {s['rank']}: 《{s['doc_name']}》·{s['section']}  score={s['score']}")
    print(f"回答      : {answer[:200]}")
    print(f"耗时      : {(time.perf_counter() - started) * 1000:.0f} ms, provider={done.get('provider')}")

    if not answer.strip():
        print("失败：回答为空")
        return 1
    if counts.get("sources", 0) == 0 or counts.get("done", 0) == 0:
        print("失败：SSE 事件不完整")
        return 1
    print("端到端链路正常")
    return 0


if __name__ == "__main__":
    sys.exit(main())
