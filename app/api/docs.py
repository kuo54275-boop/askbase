"""文档管理接口。"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile

from ..ingest.loaders import ParsingError
from ..models import DocMeta, UploadResult

router = APIRouter(tags=["docs"])

MAX_FILE_MB = 20


def _service(request: Request):
    return request.app.state.service


def _repo(request: Request):
    return request.app.state.repo


@router.post("/docs/upload", response_model=UploadResult, summary="上传文档并入库")
async def upload_doc(request: Request, file: UploadFile) -> UploadResult:
    service = _service(request)
    data = await file.read()
    if len(data) > MAX_FILE_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"文件超过 {MAX_FILE_MB}MB 限制")
    try:
        doc, preview, replaced = service.ingest(file.filename or "unnamed.txt", data)
    except ParsingError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return UploadResult(doc=DocMeta(**doc), preview=preview, replaced=replaced)


@router.get("/docs", response_model=list[DocMeta], summary="文档列表")
def list_docs(request: Request) -> list[DocMeta]:
    return _repo(request).list_documents()


@router.delete("/docs/{doc_id}", summary="删除文档")
def delete_doc(request: Request, doc_id: str) -> dict:
    service = _service(request)
    ok = _repo(request).delete_document(doc_id)
    if not ok:
        raise HTTPException(status_code=404, detail="文档不存在")
    service.refresh_index()
    return {"deleted": doc_id}
