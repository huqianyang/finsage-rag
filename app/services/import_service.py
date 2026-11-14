"""导入服务：接收上传 → 登记 documents / import_tasks → 在后台线程执行导入流程图。

上传接口只负责保存文件和登记任务，立即返回 task_id；真正耗时的解析、向量化等
在后台线程池中执行，前端通过任务进度接口轮询。任何一个文件导入失败，
只会把该任务标记为失败，不影响其他文件和整个服务（PRD"导入失败不影响整体服务"）。
"""
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
ALLOWED_EXT = {"pdf", "doc", "docx", "md"}  # 支持的文件扩展名


@lru_cache
def _executor() -> ThreadPoolExecutor:
    """导入任务线程池（全局唯一）。并发数由 IMPORT_WORKERS 控制，避免同时大量调用 MinerU 和 GPU。"""
    return ThreadPoolExecutor(max_workers=get_settings().import_workers, thread_name_prefix="import")


def _safe_name(file_name: str) -> str:
    """清理文件名：只保留最后一段，去掉路径分隔符，防止 "../../xx" 之类的名字写到临时目录之外。"""
    name = Path(file_name).name.replace("\\", "_").replace("/", "_").strip()
    return name or "unnamed"


def accept_upload(file_name: str, stream: BinaryIO, content_type: str) -> dict:
    """处理单个上传文件，返回 API 设计 §1.3 中 items 的一项。

    步骤：校验扩展名 → 保存到临时目录（同时校验大小、计算 MD5）→ 判断是否同名覆盖
    → 原文件上传 MinIO → 登记 documents 和 import_tasks → 提交后台导入。

    任何问题都以 {"accepted": False, "code", "message"} 返回而不是抛异常，
    这样一次上传多个文件时，单个文件失败不影响其他文件。
    """
    s = get_settings()
    file_name = _safe_name(file_name)
    ext = Path(file_name).suffix.lower().lstrip(".")
    if ext not in ALLOWED_EXT:
        return {"file_name": file_name, "accepted": False, "code": "UNSUPPORTED_FILE_TYPE",
                "message": "仅支持 PDF、DOC、DOCX、MD 文件"}

    # 每个任务一个独立的临时目录：temp/{task_id}/，导入结束后整体删除
    task_id = uuid.uuid4().hex
    work_dir = s.temp_dir / task_id
    work_dir.mkdir(parents=True)
    local_path = work_dir / file_name

    # 1. 分块（每次 1MB）写入本地，边写边累计大小和 MD5；超过上限立即停止并清理
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

    # 2. 同名判断：已存在同名资料 → 覆盖导入，沿用原 doc_id；正在导入中 → 拒绝，避免两个任务互相覆盖
    existing = mongo.documents().find_one({"file_name": file_name})
    if existing and existing.get("status") == "processing":
        shutil.rmtree(work_dir, ignore_errors=True)
        return {"file_name": file_name, "accepted": False, "code": "DOCUMENT_IMPORTING",
                "message": "同名文件正在导入中，请稍后再试"}
    doc_id = existing["_id"] if existing else uuid.uuid4().hex
    overwrite = existing is not None

    # 3. 原文件上传 MinIO（覆盖导入时先删掉旧的原文件和图片）
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

    # 4. 登记资料与任务
    ts = now()
    # upsert：新资料则插入；已有资料则更新。$setOnInsert 中的字段只在插入时写入
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

    # 5. 提交到后台线程池执行，接口立即返回
    state = {"task_id": task_id, "doc_id": doc_id, "file_name": file_name, "file_ext": ext,
             "local_path": str(local_path), "work_dir": str(work_dir), "content_type": content_type,
             "source_path": source_path}
    _executor().submit(_run, state)
    return {"file_name": file_name, "accepted": True, "doc_id": doc_id, "task_id": task_id, "overwrite": overwrite}


def validate_content_type(content_type: str) -> None:
    """校验资料类型代码，非法时抛 400 错误。"""
    if not is_valid(content_type):
        raise AppError("INVALID_ARGUMENT", "请选择有效的资料类型")


def _run(state: dict) -> None:
    """后台线程中执行一次导入：运行导入流程图，并把最终结果写回任务和资料记录。"""
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
        # 失败：记录失败步骤和原因，剩余步骤标记为跳过
        step = e.step if isinstance(e, StepError) else None
        message = e.message if isinstance(e, StepError) else f"导入失败：{e}"
        logger.error("导入失败 %s：%s", state["file_name"], message)
        mark_remaining_skipped(task_id)
        mongo.import_tasks().update_one({"_id": task_id}, {"$set": {
            "status": "failed", "error": {"step": step, "message": message}, "finished_at": now()}})
        mongo.documents().update_one({"_id": doc_id}, {"$set": {"status": "failed", "updated_at": now()}})
    finally:
        # 无论成功失败都清理临时目录（上传文件和 MinerU 解析结果）
        shutil.rmtree(state["work_dir"], ignore_errors=True)


def recover_interrupted() -> int:
    """服务启动时，把残留的 pending/running 任务置为失败（数据库设计 §3.2）。

    任务在内存线程池中执行，服务重启后这些任务实际上已经中断，
    如果不处理，导入页会一直显示"导入中"，同名文件也无法重新上传。返回处理的任务数。
    同时删除这些任务遗留的临时工作目录 temp/{task_id}（正常结束时由 _run 的 finally 删除，中断时没机会删）。
    """
    temp_dir = get_settings().temp_dir
    stale = list(mongo.import_tasks().find({"status": {"$in": ["pending", "running"]}}, {"_id": 1, "doc_id": 1}))
    for t in stale:
        shutil.rmtree(temp_dir / t["_id"], ignore_errors=True)
        mark_remaining_skipped(t["_id"])
        mongo.import_tasks().update_one({"_id": t["_id"]}, {"$set": {
            "status": "failed", "error": {"step": None, "message": "服务重启，导入中断，请重新上传"},
            "finished_at": now()}})
        mongo.documents().update_one({"_id": t["doc_id"], "status": "processing"},
                                     {"$set": {"status": "failed", "updated_at": now()}})
    return len(stale)


# ==================== 查询 ====================

def _task_out(t: dict) -> dict:
    """把 MongoDB 文档转成接口输出格式：_id 改名为 task_id。"""
    t = dict(t)
    t["task_id"] = t.pop("_id")
    return t


def get_task(task_id: str) -> dict:
    """查询单个导入任务（导入页轮询进度用）。"""
    t = mongo.import_tasks().find_one({"_id": task_id})
    if not t:
        raise NotFoundError("导入任务不存在")
    return _task_out(t)


def list_tasks(status: str | None, limit: int) -> list[dict]:
    """查询导入任务列表，按创建时间倒序。status="running" 表示进行中（含排队中的 pending）。"""
    q = {}
    if status == "running":
        q["status"] = {"$in": ["pending", "running"]}
    elif status:
        q["status"] = status
    return [_task_out(t) for t in mongo.import_tasks().find(q).sort("created_at", DESCENDING).limit(limit)]


def list_documents(content_type: str | None, keyword: str | None, page: int, size: int) -> dict:
    """分页查询已导入资料，可按类型筛选、按文件名模糊搜索（不区分大小写）。"""
    import re

    q: dict = {}
    if content_type:
        q["content_type"] = content_type
    if keyword:
        # re.escape：把用户输入当作普通文本，避免其中的 . * ( 等被当成正则语法
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
