"""导入任务进度记录：写 MongoDB 的 import_tasks.steps（数据库设计文档 §3.2）。

导入页每 2 秒轮询一次任务，读到的就是这里写入的步骤状态。
每个导入节点用 `@step("编码")` 装饰后，开始、完成、失败都会自动记录，节点本身不用关心进度。
"""
import functools
import logging
from datetime import UTC, datetime

from app.common.errors import StepError
from app.infra import mongo

logger = logging.getLogger(__name__)

# 导入流程的 6 个步骤：(编码, 中文名称)，顺序即执行顺序，也是导入页进度格子的显示顺序
STEPS: list[tuple[str, str]] = [
    ("parse", "解析文档"),
    ("image", "处理图片"),
    ("split", "切分文档"),
    ("metadata", "抽取元数据"),
    ("embed", "生成向量"),
    ("index", "写入向量库"),
]


def now() -> datetime:
    """当前 UTC 时间。所有写入 MongoDB 的时间统一用 UTC，前端负责换算成本地时间。"""
    return datetime.now(UTC)


def initial_steps() -> list[dict]:
    """新建任务时的步骤列表：全部为 pending（等待中）。"""
    return [{"code": c, "label": label, "status": "pending", "started_at": None, "finished_at": None, "message": ""}
            for c, label in STEPS]


def _set_step(task_id: str, code: str, **fields) -> None:
    """更新任务中某一个步骤的若干字段。

    使用 MongoDB 的 arrayFilters：`steps.$[s].status` 表示"steps 数组中满足条件 s 的元素的 status"，
    条件 s 为 `s.code == code`，这样无需知道该步骤在数组中的下标。
    """
    update = {f"steps.$[s].{k}": v for k, v in fields.items()}
    mongo.import_tasks().update_one({"_id": task_id}, {"$set": update}, array_filters=[{"s.code": code}])


def step(code: str):
    """节点装饰器：自动记录步骤开始/完成/失败。

    - 开始：步骤状态置为 running，并记录任务当前步骤 current_step
    - 完成：状态置为 done；节点可在返回值中放 `_message`（如"共 18 个切片"）作为步骤说明，
      装饰器会把它取出写入进度，不会进入 LangGraph 状态
    - 失败：状态置为 failed；非 StepError 的异常会包装成 StepError，保证错误信息面向用户
    """

    def deco(fn):
        @functools.wraps(fn)
        def wrapper(state):
            task_id = state["task_id"]
            mongo.import_tasks().update_one({"_id": task_id}, {"$set": {"current_step": code}})
            _set_step(task_id, code, status="running", started_at=now())
            try:
                result = fn(state) or {}
            except StepError:
                # 节点自己抛出的业务错误，信息已经是给用户看的，直接上抛
                _set_step(task_id, code, status="failed", finished_at=now())
                raise
            except Exception as e:
                # 未预料的异常：记录完整堆栈便于排查，再包装成"某某步骤失败：原因"
                logger.exception("导入步骤 %s 失败", code)
                _set_step(task_id, code, status="failed", finished_at=now())
                raise StepError(code, f"{dict(STEPS)[code]}失败：{e}") from e
            message = result.pop("_message", "")
            _set_step(task_id, code, status="done", finished_at=now(), message=message)
            return result

        return wrapper

    return deco


def mark_remaining_skipped(task_id: str) -> None:
    """任务失败后，把还没执行的步骤（pending）标记为 skipped，导入页上显示为删除线。"""
    task = mongo.import_tasks().find_one({"_id": task_id}, {"steps": 1})
    for s in (task or {}).get("steps", []):
        if s["status"] == "pending":
            _set_step(task_id, s["code"], status="skipped")
