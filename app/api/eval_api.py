"""评测接口。"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from ..config import BASE_DIR
from ..evaluation import evaluate, load_cases
from ..models import EvalResult

router = APIRouter(tags=["eval"])

CASES_PATH = BASE_DIR / "eval" / "qa_set.jsonl"


@router.post("/eval/run", response_model=EvalResult, summary="跑离线评测（不消耗 token）")
def run_eval(request: Request) -> EvalResult:
    if not CASES_PATH.exists():
        raise HTTPException(status_code=404, detail=f"未找到标注集 {CASES_PATH}")
    cases = load_cases(CASES_PATH)
    if not cases:
        raise HTTPException(status_code=400, detail="标注集为空")
    return evaluate(request.app.state.service, cases)
