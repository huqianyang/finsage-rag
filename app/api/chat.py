"""查询服务路由（API 设计 §2）。"""
from fastapi import APIRouter, Header, Query
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import RedirectResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

from app.common.errors import AppError, NotFoundError
from app.infra import mongo, object_store
from app.services import chat_service, session_service

router = APIRouter(prefix="/api")


def _client(x_client_id: str | None) -> str:
    if not x_client_id or len(x_client_id) > 64:
        raise AppError("INVALID_ARGUMENT", "缺少有效的 X-Client-Id 请求头")
    return x_client_id


class ChatRequest(BaseModel):
    query: str | None = Field(None, max_length=500)
    session_id: str | None = None
    selected_doc_id: str | None = None
    regenerate_message_id: str | None = None


@router.post("/chat/stream")
async def chat_stream(req: ChatRequest, x_client_id: str | None = Header(None)):
    client_id = _client(x_client_id)
    ctx = await run_in_threadpool(chat_service.prepare, client_id, req.query, req.session_id,
                                  req.selected_doc_id, req.regenerate_message_id)
    return StreamingResponse(chat_service.stream_response(ctx), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@router.get("/sessions")
def list_sessions(x_client_id: str | None = Header(None), limit: int = Query(50, ge=1, le=200)):
    return {"items": session_service.list_sessions(_client(x_client_id), limit)}


@router.post("/sessions", status_code=201)
def create_session(x_client_id: str | None = Header(None)):
    return session_service.create_session(_client(x_client_id))


@router.get("/sessions/{session_id}/messages")
def list_messages(session_id: str, x_client_id: str | None = Header(None)):
    return {"session_id": session_id, "items": session_service.list_messages(_client(x_client_id), session_id)}


@router.delete("/sessions/{session_id}", status_code=204)
def delete_session(session_id: str, x_client_id: str | None = Header(None)):
    session_service.delete_session(_client(x_client_id), session_id)
    return Response(status_code=204)


@router.get("/documents/{doc_id}/file")
def open_file(doc_id: str):
    doc = mongo.documents().find_one({"_id": doc_id}, {"source_path": 1})
    if not doc:
        raise NotFoundError("资料不存在")
    return RedirectResponse(object_store.presigned_url(doc["source_path"]), status_code=302)
