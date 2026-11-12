"""导入服务：接收上传 → 登记 documents / import_tasks → 后台线程执行导入图。"""
import hashlib
import logging
import shutil
import uuid
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
from typing import BinaryIO

from pymongo import DESCENDING

from app.common.content_types import is_valid, label_of
from app.common.errors import AppError, NotFoundError, StepError
from app.config import get_settings
from app.infra import mongo, object_store
from app.pipeline.importing.graph import build_import_graph
from app.pipeline.importing.progress import initial_steps, mark_remaining_skipped, now

logger = logging.getLogger(__name__)
ALLOWED_EXT = {"pdf", "doc", "docx", "md"}


@lru_cache
def _executor() -> ThreadPoolExecutor:
    return ThreadPoolExecutor(max_workers=get_settings().import_workers, thread_name_prefix="import")


def _safe_name(file_name: str) -> str:
    name = Path(file_name).name.replace("\\", "_").replace("/", "_").strip()
    return name or "unnamed"


def accept_upload(file_name: str, stream: BinaryIO, content_type: str) -> dict:
    """处理单个文件，返回 API 设计 §1.3 中 items 的一项。异常以 accepted=false 返回，不影响其他文件。"""
    s = get_settings()
    file_name = _safe_name(file_name)
    ext = Path(file_name).suffix.lower().lstrip(".")
    if ext not in ALLOWED_EXT:
        return {"file_name": file_name, "accepted": False, "code": "UNSUPPORTED_FILE_TYPE",
                "message": "仅支持 PDF、DOC、DOCX、MD 文件"}

    task_id = uuid.uuid4().hex
    work_dir = s.temp_dir / task_id
    work_dir.mkdir(parents=True)
    local_path = work_dir / file_name

    # 1. 落盘 + 校验大小 + md5
    md5, size, limit = hashlib.md5(), 0, s.upload_max_mb * 1024 * 1024
    with local_path.open("wb") as f:
        while chunk := stream.read(1024 * 1024):
            size += len(chunk)
            if size > limit:
                f.close()
                shutil.rmtree(work_dir, ignore_errors=True)
                return {"file_name": file_name, "accepted": False, "code": "FILE_TOO_LARGE",
                        "message": f"文件超过 {s.upload_max_mb}MB"}
            md5.update(chunk)
            f.write(chunk)
    if size == 0:
        shutil.rmtree(work_dir, ignore_errors=True)
        return {"file_name": file_name, "accepted": False, "code": "INVALID_ARGUMENT", "message": "文件为空"}

    # 2. 同名判断（覆盖导入复用 doc_id）
    existing = mongo.documents().find_one({"file_name": file_name})
    if existing and existing.get("status") == "processing":
        shutil.rmtree(work_dir, ignore_errors=True)
        return {"file_name": file_name, "accepted": False, "code": "DOCUMENT_IMPORTING",
                "message": "同名文件正在导入中，请稍后再试"}
    doc_id = existing["_id"] if existing else uuid.uuid4().hex
    overwrite = existing is not None

    # 3. 原文件上传 MinIO
    source_path = object_store.raw_object_name(doc_id, file_name)
    try:
        if overwrite:
            object_store.remove_prefix(f"raw/{doc_id}/")
            object_store.remove_prefix(f"images/{doc_id}/")
        object_store.upload_file(source_path, local_path)
    except Exception as e:
        logger.exception("上传 MinIO 失败")
        shutil.rmtree(work_dir, ignore_errors=True)
        return {"file_name": file_name, "accepted": False, "code": "DEPENDENCY_UNAVAILABLE",
                "message": f"文件存储失败：{e}"}

    # 4. 登记
    ts = now()
    mongo.documents().update_one({"_id": doc_id}, {
        "$set": {"file_name": file_name, "file_ext": ext, "file_size": size, "file_md5": md5.hexdigest(),
                 "content_type": content_type, "source_path": source_path, "status": "processing",
                 "latest_task_id": task_id, "updated_at": ts},
        "$setOnInsert": {"created_at": ts, "chunk_count": 0, "metadata": {}},
    }, upsert=True)
    mongo.import_tasks().insert_one({
        "_id": task_id, "doc_id": doc_id, "file_name": file_name, "status": "pending", "current_step": None,
        "steps": initial_steps(), "error": None, "created_at": ts, "started_at": None, "finished_at": None,
    })

    state = {"task_id": task_id, "doc_id": doc_id, "file_name": file_name, "file_ext": ext,
             "local_path": str(local_path), "work_dir": str(work_dir), "content_type": content_type,
             "source_path": source_path}
    _executor().submit(_run, state)
    return {"file_name": file_name, "accepted": True, "doc_id": doc_id, "task_id": task_id, "overwrite": overwrite}


def validate_content_type(content_type: str) -> None:
    if not is_valid(content_type):
        raise AppError("INVALID_ARGUMENT", "请选择有效的资料类型")


def _run(state: dict) -> None:
    task_id, doc_id = state["task_id"], state["doc_id"]
    mongo.import_tasks().update_one({"_id": task_id}, {"$set": {"status": "running", "started_at": now()}})
    try:
        final = build_import_graph().invoke(state)
        mongo.import_tasks().update_one({"_id": task_id}, {"$set": {
            "status": "completed", "current_step": None, "finished_at": now()}})
        mongo.documents().update_one({"_id": doc_id}, {"$set": {
            "status": "completed", "chunk_count": len(final.get("chunks", [])),
            "metadata": final.get("metadata", {}), "updated_at": now()}})
        logger.info("导入完成 %s", state["file_name"])
    except Exception as e:
        step = e.step if isinstance(e, StepError) else None
        message = e.message if isinstance(e, StepError) else f"导入失败：{e}"
        logger.error("导入失败 %s：%s", state["file_name"], message)
        mark_remaining_skipped(task_id)
        mongo.import_tasks().update_one({"_id": task_id}, {"$set": {
            "status": "failed", "error": {"step": step, "message": message}, "finished_at": now()}})
        mongo.documents().update_one({"_id": doc_id}, {"$set": {"status": "failed", "updated_at": now()}})
    finally:
        shutil.rmtree(state["work_dir"], ignore_errors=True)


def recover_interrupted() -> int:
    """服务启动时，把残留的 pending/running 任务置为失败（数据库设计 §3.2）。"""
    stale = list(mongo.import_tasks().find({"status": {"$in": ["pending", "running"]}}, {"_id": 1, "doc_id": 1}))
    for t in stale:
        mark_remaining_skipped(t["_id"])
        mongo.import_tasks().update_one({"_id": t["_id"]}, {"$set": {
            "status": "failed", "error": {"step": None, "message": "服务重启，导入中断，请重新上传"},
            "finished_at": now()}})
        mongo.documents().update_one({"_id": t["doc_id"], "status": "processing"},
                                     {"$set": {"status": "failed", "updated_at": now()}})
    return len(stale)


# ---------- 查询 ----------

def _task_out(t: dict) -> dict:
    t = dict(t)
    t["task_id"] = t.pop("_id")
    return t


def get_task(task_id: str) -> dict:
    t = mongo.import_tasks().find_one({"_id": task_id})
    if not t:
        raise NotFoundError("导入任务不存在")
    return _task_out(t)


def list_tasks(status: str | None, limit: int) -> list[dict]:
    q = {}
    if status == "running":
        q["status"] = {"$in": ["pending", "running"]}
    elif status:
        q["status"] = status
    return [_task_out(t) for t in mongo.import_tasks().find(q).sort("created_at", DESCENDING).limit(limit)]


def list_documents(content_type: str | None, keyword: str | None, page: int, size: int) -> dict:
    import re

    q: dict = {}
    if content_type:
        q["content_type"] = content_type
    if keyword:
        q["file_name"] = {"$regex": re.escape(keyword), "$options": "i"}
    total = mongo.documents().count_documents(q)
    cursor = mongo.documents().find(q).sort("updated_at", DESCENDING).skip((page - 1) * size).limit(size)
    items = []
    for d in cursor:
        items.append({
            "doc_id": d["_id"], "file_name": d["file_name"], "content_type": d["content_type"],
            "content_type_label": label_of(d["content_type"]), "status": d["status"],
            "chunk_count": d.get("chunk_count", 0), "metadata": d.get("metadata", {}),
            "latest_task_id": d.get("latest_task_id"), "updated_at": d.get("updated_at"),
        })
    return {"total": total, "page": page, "size": size, "items": items}
