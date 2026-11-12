"""导入服务路由（API 设计 §1）：资料类型、上传、导入进度、资料列表。"""
from fastapi import APIRouter, File, Form, Query, UploadFile
from fastapi.concurrency import run_in_threadpool

from app.common.content_types import CONTENT_TYPES
from app.common.errors import AppError
from app.services import import_service

router = APIRouter(prefix="/api")


@router.get("/content-types")
def content_types():
    """资料类型下拉选项：[{code, label}, ...]。"""
    return [{"code": k, "label": v} for k, v in CONTENT_TYPES.items()]


@router.post("/documents", status_code=202)
async def upload(files: list[UploadFile] = File(...), content_type: str = Form(...)):
    """上传一个或多个文件（multipart/form-data），本次上传的文件共用一个资料类型。

    返回 202 Accepted：文件已接收，导入在后台进行；每个文件单独返回是否被接收及对应的 task_id。
    """
    import_service.validate_content_type(content_type)
    if not files:
        raise AppError("INVALID_ARGUMENT", "请至少选择一个文件")
    items = []
    for f in files:
        # accept_upload 有磁盘读写和 MinIO 上传等阻塞操作，放到线程池执行，避免卡住事件循环
        items.append(await run_in_threadpool(import_service.accept_upload, f.filename or "", f.file, content_type))
    return {"items": items}


@router.get("/import-tasks/{task_id}")
def get_task(task_id: str):
    """查询单个导入任务的进度（导入页每 2 秒轮询一次）。"""
    return import_service.get_task(task_id)


@router.get("/import-tasks")
def list_tasks(status: str | None = None, limit: int = Query(20, ge=1, le=100)):
    """导入任务列表；status=running 时返回进行中的任务（导入页刷新后用于恢复进度显示）。"""
    return {"items": import_service.list_tasks(status, limit)}


@router.get("/documents")
def list_documents(content_type: str | None = None, keyword: str | None = None,
                   page: int = Query(1, ge=1), size: int = Query(20, ge=1, le=100)):
    """已导入资料列表：可按类型筛选、按文件名搜索，分页返回。"""
    return import_service.list_documents(content_type, keyword, page, size)
