"""导入服务路由（API 设计 §1）。"""
from fastapi import APIRouter, File, Form, Query, UploadFile
from fastapi.concurrency import run_in_threadpool

from app.common.content_types import CONTENT_TYPES
from app.common.errors import AppError
from app.services import import_service

router = APIRouter(prefix="/api")


@router.get("/content-types")
def content_types():
    return [{"code": k, "label": v} for k, v in CONTENT_TYPES.items()]


@router.post("/documents", status_code=202)
async def upload(files: list[UploadFile] = File(...), content_type: str = Form(...)):
    import_service.validate_content_type(content_type)
    if not files:
        raise AppError("INVALID_ARGUMENT", "请至少选择一个文件")
    items = []
    for f in files:
        items.append(await run_in_threadpool(import_service.accept_upload, f.filename or "", f.file, content_type))
    return {"items": items}


@router.get("/import-tasks/{task_id}")
def get_task(task_id: str):
    return import_service.get_task(task_id)


@router.get("/import-tasks")
def list_tasks(status: str | None = None, limit: int = Query(20, ge=1, le=100)):
    return {"items": import_service.list_tasks(status, limit)}


@router.get("/documents")
def list_documents(content_type: str | None = None, keyword: str | None = None,
                   page: int = Query(1, ge=1), size: int = Query(20, ge=1, le=100)):
    return import_service.list_documents(content_type, keyword, page, size)
