"""查询服务路由（API 设计 §2）：流式问答、会话管理、查看原文。

会话相关接口都要求请求头 X-Client-Id（浏览器匿名 ID），用于区分不同用户的会话。
"""
from fastapi import APIRouter, Header, Query
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import RedirectResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

from app.common.errors import AppError, NotFoundError
from app.infra import mongo, object_store
from app.services import chat_service, session_service

router = APIRouter(prefix="/api")


def _client(x_client_id: str | None) -> str:
    """校验 X-Client-Id 请求头（必填，不超过 64 个字符）。"""
    if not x_client_id or len(x_client_id) > 64:
        raise AppError("INVALID_ARGUMENT", "缺少有效的 X-Client-Id 请求头")
    return x_client_id


class ChatRequest(BaseModel):
    """提问请求体。"""
    query: str | None = Field(None, max_length=500)  # 问题；重新回答时可省略
    session_id: str | None = None  # 会话 ID；为空时自动新建会话
    selected_doc_id: str | None = None  # 澄清时用户选中的资料 ID
    regenerate_message_id: str | None = None  # 要重新回答的助手消息 ID


@router.post("/chat/stream")
async def chat_stream(req: ChatRequest, x_client_id: str | None = Header(None)):
    """提问并以 SSE 流式返回回答（Content-Type: text/event-stream）。

    参数校验和写入用户消息在建立流之前完成：出错时直接返回普通的 JSON 错误响应，
    前端更容易处理；校验通过后才开始返回事件流。
    """
    client_id = _client(x_client_id)
    ctx = await run_in_threadpool(chat_service.prepare, client_id, req.query, req.session_id,
                                  req.selected_doc_id, req.regenerate_message_id)
    # Cache-Control: no-cache 禁止缓存；X-Accel-Buffering: no 告诉 Nginx 等代理不要缓冲，保证逐条实时送达
    return StreamingResponse(chat_service.stream_response(ctx), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@router.get("/sessions")
def list_sessions(x_client_id: str | None = Header(None), limit: int = Query(50, ge=1, le=200)):
    """当前浏览器的会话列表（最近更新的在前）。"""
    return {"items": session_service.list_sessions(_client(x_client_id), limit)}


@router.post("/sessions", status_code=201)
def create_session(x_client_id: str | None = Header(None)):
    """新建空会话（"新建会话"按钮使用）。"""
    return session_service.create_session(_client(x_client_id))


@router.get("/sessions/{session_id}/messages")
def list_messages(session_id: str, x_client_id: str | None = Header(None)):
    """会话的全部消息（按时间正序），用于打开会话或刷新页面后恢复聊天记录。"""
    return {"session_id": session_id, "items": session_service.list_messages(_client(x_client_id), session_id)}


@router.delete("/sessions/{session_id}", status_code=204)
def delete_session(session_id: str, x_client_id: str | None = Header(None)):
    """删除会话及其全部消息，返回 204（无响应体）。"""
    session_service.delete_session(_client(x_client_id), session_id)
    return Response(status_code=204)


@router.get("/documents/{doc_id}/file")
def open_file(doc_id: str):
    """查看原文：生成 MinIO 临时签名链接（1 小时有效）并 302 跳转过去，引用卡片的"查看原文"使用。

    MinIO 桶是私有的，浏览器不能直接访问对象；每次点击都生成新链接，不会出现链接过期的问题。
    """
    doc = mongo.documents().find_one({"_id": doc_id}, {"source_path": 1})
    if not doc:
        raise NotFoundError("资料不存在")
    return RedirectResponse(object_store.presigned_url(doc["source_path"]), status_code=302)
