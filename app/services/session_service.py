"""会话与消息（数据库设计 §3.3 / §3.4，API 设计 §2.2–2.5）。"""
import uuid

from pymongo import ASCENDING, DESCENDING

from app.common.errors import NotFoundError
from app.infra import mongo
from app.pipeline.importing.progress import now

DEFAULT_TITLE = "新会话"


def _session_out(s: dict) -> dict:
    return {"session_id": s["_id"], "title": s["title"], "message_count": s.get("message_count", 0),
            "locked_entity": s.get("locked_entity"), "updated_at": s["updated_at"]}


def create_session(client_id: str) -> dict:
    ts = now()
    doc = {"_id": uuid.uuid4().hex, "client_id": client_id, "title": DEFAULT_TITLE, "locked_entity": None,
           "message_count": 0, "created_at": ts, "updated_at": ts}
    mongo.sessions().insert_one(doc)
    return _session_out(doc)


def get_owned(client_id: str, session_id: str) -> dict:
    s = mongo.sessions().find_one({"_id": session_id, "client_id": client_id})
    if not s:
        raise NotFoundError("该会话不存在或已被删除")
    return s


def list_sessions(client_id: str, limit: int) -> list[dict]:
    cur = mongo.sessions().find({"client_id": client_id}).sort("updated_at", DESCENDING).limit(limit)
    return [_session_out(s) for s in cur]


def delete_session(client_id: str, session_id: str) -> None:
    get_owned(client_id, session_id)
    mongo.messages().delete_many({"session_id": session_id})
    mongo.sessions().delete_one({"_id": session_id})


def list_messages(client_id: str, session_id: str) -> list[dict]:
    get_owned(client_id, session_id)
    out = []
    for m in mongo.messages().find({"session_id": session_id}, {"trace": 0}).sort("created_at", ASCENDING):
        m["message_id"] = m.pop("_id")
        out.append(m)
    return out


def recent_history(session_id: str, before_message_id: str | None, turns: int) -> list[dict[str, str]]:
    """取当前问题之前最近 N 轮已完成的问答，用于问题改写与生成。"""
    msgs = list(mongo.messages().find(
        {"session_id": session_id, "status": {"$in": ["completed", "refused", "no_answer"]}},
        {"role": 1, "content": 1, "created_at": 1},
    ).sort("created_at", ASCENDING))
    if before_message_id:
        ids = [m["_id"] for m in msgs]
        if before_message_id in ids:
            msgs = msgs[: ids.index(before_message_id)]
    return [{"role": m["role"], "content": m["content"]} for m in msgs][-turns * 2:]
