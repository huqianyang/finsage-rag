"""会话与消息（数据库设计 §3.3 / §3.4，API 设计 §2.2–2.5）。

本项目不做登录：浏览器首次访问时生成一个匿名 client_id 存在 localStorage，
每个请求通过 X-Client-Id 请求头带上。会话归属于 client_id，
访问不属于自己的会话一律按"不存在"处理（返回 404），不暴露会话是否存在。
"""
import uuid

from pymongo import ASCENDING, DESCENDING

from app.common.errors import NotFoundError
from app.infra import mongo
from app.pipeline.importing.progress import now

DEFAULT_TITLE = "新会话"  # 会话的初始标题；第一个问题提出后改为问题前 30 字


def _session_out(s: dict) -> dict:
    """把 MongoDB 中的会话文档转成接口输出格式（_id → session_id，不返回 client_id）。"""
    return {"session_id": s["_id"], "title": s["title"], "message_count": s.get("message_count", 0),
            "locked_entity": s.get("locked_entity"), "updated_at": s["updated_at"]}


def create_session(client_id: str) -> dict:
    """新建一个空会话。"""
    ts = now()
    doc = {"_id": uuid.uuid4().hex, "client_id": client_id, "title": DEFAULT_TITLE, "locked_entity": None,
           "message_count": 0, "created_at": ts, "updated_at": ts}
    mongo.sessions().insert_one(doc)
    return _session_out(doc)


def get_owned(client_id: str, session_id: str) -> dict:
    """获取属于该 client 的会话；不存在或不属于该 client 时抛 404。"""
    s = mongo.sessions().find_one({"_id": session_id, "client_id": client_id})
    if not s:
        raise NotFoundError("该会话不存在或已被删除")
    return s


def list_sessions(client_id: str, limit: int) -> list[dict]:
    """某个浏览器的会话列表，最近更新的排在前面。"""
    cur = mongo.sessions().find({"client_id": client_id}).sort("updated_at", DESCENDING).limit(limit)
    return [_session_out(s) for s in cur]


def delete_session(client_id: str, session_id: str) -> None:
    """删除会话及其全部消息（物理删除，不可恢复）。"""
    get_owned(client_id, session_id)  # 先校验归属
    mongo.messages().delete_many({"session_id": session_id})
    mongo.sessions().delete_one({"_id": session_id})


def list_messages(client_id: str, session_id: str) -> list[dict]:
    """会话的全部消息，按时间正序；不返回 trace（检索轨迹仅用于排查，体积也较大）。"""
    get_owned(client_id, session_id)
    out = []
    for m in mongo.messages().find({"session_id": session_id}, {"trace": 0}).sort("created_at", ASCENDING):
        m["message_id"] = m.pop("_id")
        out.append(m)
    return out


def recent_history(session_id: str, before_message_id: str | None, turns: int) -> list[dict[str, str]]:
    """取当前问题之前最近 N 轮已完成的问答，用于问题改写与生成。

    - 只取正常结束的消息（completed / refused / no_answer），跳过澄清、已停止、失败的消息；
    - before_message_id 为当前问题的消息 ID：只取它之前的消息
      （重新回答时，当前问题后面可能还有其他消息，不能算进历史）；
    - 一轮 = 一问一答，所以取最后 turns * 2 条。
    """
    msgs = list(mongo.messages().find(
        {"session_id": session_id, "status": {"$in": ["completed", "refused", "no_answer"]}},
        {"role": 1, "content": 1, "created_at": 1},
    ).sort("created_at", ASCENDING))
    if before_message_id:
        ids = [m["_id"] for m in msgs]
        if before_message_id in ids:
            msgs = msgs[: ids.index(before_message_id)]
    return [{"role": m["role"], "content": m["content"]} for m in msgs][-turns * 2:]
