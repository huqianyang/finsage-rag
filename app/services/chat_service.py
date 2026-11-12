"""问答服务：组织一次提问的完整生命周期，并把查询图的事件转成 SSE（API 设计 §2.1）。"""
import json
import logging
import time
import uuid
from collections.abc import Iterator

from openai import APITimeoutError

from app.common.errors import AppError, NotFoundError
from app.config import get_settings
from app.infra import mongo
from app.pipeline.importing.progress import now
from app.pipeline.querying.graph import build_query_graph
from app.services import session_service

logger = logging.getLogger(__name__)


def sse(event: str, data) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False, default=str)}\n\n"


def prepare(client_id: str, query: str | None, session_id: str | None, selected_doc_id: str | None,
            regenerate_message_id: str | None) -> dict:
    """同步校验并写入用户消息；出错时抛 AppError（在建立 SSE 之前返回普通 HTTP 错误）。"""
    if session_id:
        session = session_service.get_owned(client_id, session_id)
    else:
        created = session_service.create_session(client_id)
        session = session_service.get_owned(client_id, created["session_id"])

    if regenerate_message_id:
        old = mongo.messages().find_one({"_id": regenerate_message_id, "session_id": session["_id"],
                                         "role": "assistant"})
        if not old:
            raise NotFoundError("要重新回答的消息不存在")
        user_msg = mongo.messages().find_one({"_id": old["reply_to"]})
        if not user_msg:
            raise NotFoundError("原问题不存在")
        mongo.messages().delete_one({"_id": regenerate_message_id})
        mongo.sessions().update_one({"_id": session["_id"]}, {"$inc": {"message_count": -1}})
        query = user_msg["content"]
        user_message_id = user_msg["_id"]
    else:
        query = (query or "").strip()
        if not query:
            raise AppError("INVALID_ARGUMENT", "问题不能为空")
        if len(query) > 500:
            raise AppError("INVALID_ARGUMENT", "问题请控制在 500 字以内")
        user_message_id = uuid.uuid4().hex
        mongo.messages().insert_one({"_id": user_message_id, "session_id": session["_id"], "role": "user",
                                     "content": query, "status": "completed", "created_at": now()})
        updates = {"$inc": {"message_count": 1}, "$set": {"updated_at": now()}}
        if session.get("message_count", 0) == 0 or session.get("title") == session_service.DEFAULT_TITLE:
            updates["$set"]["title"] = query[:30]
        mongo.sessions().update_one({"_id": session["_id"]}, updates)

    return {"session": session, "query": query, "user_message_id": user_message_id,
            "assistant_message_id": uuid.uuid4().hex, "selected_doc_id": selected_doc_id}


def run_stream(ctx: dict) -> Iterator[str]:
    session, s = ctx["session"], get_settings()
    t0 = time.perf_counter()
    yield sse("start", {"session_id": session["_id"], "user_message_id": ctx["user_message_id"],
                        "assistant_message_id": ctx["assistant_message_id"]})

    state_in = {
        "query": ctx["query"],
        "history": session_service.recent_history(session["_id"], ctx["user_message_id"], s.history_turns),
        "locked_entity": session.get("locked_entity"),
        "selected_doc_id": ctx["selected_doc_id"],
    }
    final: dict = {}
    partial = ""
    status = "failed"
    try:
        for mode, payload in build_query_graph().stream(state_in, stream_mode=["custom", "values"]):
            if mode == "values":
                final = payload
                continue
            event, data = payload
            if event == "delta":
                partial += data["text"]
            elif event == "replace":
                partial = partial[:data["start"]] + data["text"] + partial[data["end"]:]
            yield sse(event, data)

        status = final.get("outcome", "completed")
        appendix = final.get("appendix") or {}
        if status != "clarify":
            yield sse("appendix", {"risk_notice": appendix.get("risk_notice"),
                                   "realtime_notice": appendix.get("realtime_notice")})
            yield sse("citations", final.get("citations", []))
        elapsed = int((time.perf_counter() - t0) * 1000)
        yield sse("done", {"status": status, "elapsed_ms": elapsed,
                           "rewritten_query": final.get("rewritten_query"),
                           "entity": final.get("entity") or session.get("locked_entity")})
    except GeneratorExit:  # 客户端断开 = 停止生成
        status = "stopped"
        raise
    except APITimeoutError as e:
        logger.warning("LLM 超时：%s", e)
        yield sse("error", {"code": "LLM_TIMEOUT", "message": "回答生成失败：模型服务响应超时，请稍后重试。"})
    except Exception as e:
        logger.exception("问答失败")
        code = "LLM_ERROR" if "openai" in type(e).__module__ else "RETRIEVAL_ERROR"
        msg = "回答生成失败：模型服务暂时不可用，请稍后重试。" if code == "LLM_ERROR" else "资料检索失败，请稍后重试。"
        yield sse("error", {"code": code, "message": msg})
    finally:
        _save(ctx, final, status, partial, int((time.perf_counter() - t0) * 1000))


def _save(ctx: dict, final: dict, status: str, partial: str, elapsed: int) -> None:
    session = ctx["session"]
    content = final.get("answer") if status not in ("stopped", "failed") else partial
    if status == "failed" and not content:
        content = ""
    trace = {
        "dense": [h["chunk_id"] for h in final.get("dense_hits", [])],
        "bm25": [h["chunk_id"] for h in final.get("bm25_hits", [])],
        "web": [h.get("url") for h in final.get("web_hits", [])],
        "rerank": [{"id": x.get("chunk_id") or x.get("url"), "score": x.get("score")}
                   for x in final.get("reranked", [])],
        "timings": final.get("timings", {}),
    }
    doc = {
        "_id": ctx["assistant_message_id"], "session_id": session["_id"], "role": "assistant",
        "content": content or "", "status": status, "reply_to": ctx["user_message_id"],
        "rewritten_query": final.get("rewritten_query"), "intent": final.get("intent"),
        "entity": final.get("entity"), "clarify_candidates": final.get("clarify_candidates"),
        "citations": final.get("citations", []) if status not in ("stopped", "failed") else [],
        "appendix": final.get("appendix") if status not in ("stopped", "failed") else None,
        "trace": trace, "elapsed_ms": elapsed, "created_at": now(),
    }
    try:
        mongo.messages().insert_one(doc)
        updates: dict = {"$inc": {"message_count": 1}, "$set": {"updated_at": now()}}
        if final.get("entity"):
            updates["$set"]["locked_entity"] = final["entity"]
        mongo.sessions().update_one({"_id": session["_id"]}, updates)
    except Exception:
        logger.exception("保存回答失败")
