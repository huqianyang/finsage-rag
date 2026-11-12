"""导入任务进度：写 MongoDB import_tasks.steps（数据库设计 §3.2）。"""
import functools
import logging
from datetime import UTC, datetime

from app.common.errors import StepError
from app.infra import mongo

logger = logging.getLogger(__name__)

STEPS: list[tuple[str, str]] = [
    ("parse", "解析文档"),
    ("image", "处理图片"),
    ("split", "切分文档"),
    ("metadata", "抽取元数据"),
    ("embed", "生成向量"),
    ("index", "写入向量库"),
]


def now() -> datetime:
    return datetime.now(UTC)


def initial_steps() -> list[dict]:
    return [{"code": c, "label": label, "status": "pending", "started_at": None, "finished_at": None, "message": ""}
            for c, label in STEPS]


def _set_step(task_id: str, code: str, **fields) -> None:
    update = {f"steps.$[s].{k}": v for k, v in fields.items()}
    mongo.import_tasks().update_one({"_id": task_id}, {"$set": update}, array_filters=[{"s.code": code}])


def step(code: str):
    """节点装饰器：自动记录步骤开始/完成/失败。节点可在返回值中放 `_message` 作为步骤说明。"""

    def deco(fn):
        @functools.wraps(fn)
        def wrapper(state):
            task_id = state["task_id"]
            mongo.import_tasks().update_one({"_id": task_id}, {"$set": {"current_step": code}})
            _set_step(task_id, code, status="running", started_at=now())
            try:
                result = fn(state) or {}
            except StepError:
                _set_step(task_id, code, status="failed", finished_at=now())
                raise
            except Exception as e:
                logger.exception("导入步骤 %s 失败", code)
                _set_step(task_id, code, status="failed", finished_at=now())
                raise StepError(code, f"{dict(STEPS)[code]}失败：{e}") from e
            message = result.pop("_message", "")
            _set_step(task_id, code, status="done", finished_at=now(), message=message)
            return result

        return wrapper

    return deco


def mark_remaining_skipped(task_id: str) -> None:
    task = mongo.import_tasks().find_one({"_id": task_id}, {"steps": 1})
    for s in (task or {}).get("steps", []):
        if s["status"] == "pending":
            _set_step(task_id, s["code"], status="skipped")
