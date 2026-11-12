"""降级与兜底集成测试（PRD 验收第 5 条、G3）。

需要真实的 Milvus 与本地模型、已导入首批资料；用 monkeypatch 模拟外部服务故障。
运行：uv run pytest -m integration -q
"""
import json

import openai
import pytest

from app.compliance.rules import NO_ANSWER_TEXT
from app.infra import llm, local_models, web_search
from app.pipeline.querying import nodes
from app.pipeline.querying.graph import build_query_graph

pytestmark = pytest.mark.integration


def run(query: str, history=None) -> dict:
    final, events = {}, []
    for mode, payload in build_query_graph().stream(
            {"query": query, "history": history or [], "locked_entity": None, "selected_doc_id": None},
            stream_mode=["custom", "values"]):
        if mode == "values":
            final = payload
        else:
            events.append(payload[0])
    final["_events"] = events
    return final


def test_web_search_down_still_answers_from_kb(monkeypatch):
    async def boom(*a, **k):
        raise ConnectionError("模拟联网搜索不可用")
    monkeypatch.setattr(web_search, "_call", boom)
    r = run("华夏债券C的赎回费是多少？")
    assert r["outcome"] == "completed"
    assert r["web_hits"] == []
    assert r["citations"] and all(c["source_type"] == "kb" for c in r["citations"])


def test_reranker_down_falls_back_to_rrf(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("模拟重排模型不可用")
    monkeypatch.setattr(local_models, "rerank", boom)
    r = run("招商银行一季度营业收入是多少？")
    assert r["outcome"] == "completed"
    assert r["citations"], "重排失败后仍应按 RRF 顺序给出引用"


def test_no_kb_and_no_web_returns_fixed_text(monkeypatch):
    monkeypatch.setattr(web_search, "search", lambda q: [])
    r = run("某某银行信用卡年费怎么收？")
    assert r["outcome"] == "no_answer"
    assert r["answer"] == NO_ANSWER_TEXT
    assert r["citations"] == []


def test_stop_saves_partial_answer(monkeypatch):
    import threading

    from app.services import chat_service

    saved = {}
    monkeypatch.setattr(chat_service, "_save", lambda ctx, final, status, partial, elapsed: saved.update(
        status=status, partial=partial))
    monkeypatch.setattr(chat_service.session_service, "recent_history", lambda *a, **k: [])
    ctx = {"session": {"_id": "test-session", "locked_entity": None}, "query": "什么是ETF？",
           "user_message_id": "u", "assistant_message_id": "a", "selected_doc_id": None}
    stop = threading.Event()
    events = []
    for chunk in chat_service.run_stream(ctx, stop):
        events.append(chunk)
        if chunk.startswith("event: delta"):
            stop.set()  # 模拟客户端点击“停止”
    assert saved["status"] == "stopped"
    assert saved["partial"], "应保存已生成的部分内容"
    assert not any(e.startswith("event: done") for e in events)


def test_llm_down_returns_error_event(monkeypatch):
    from app.services import chat_service

    def boom(*a, **k):
        raise openai.APIConnectionError(request=None)
    monkeypatch.setattr(llm, "chat_stream", boom)
    monkeypatch.setattr(chat_service, "_save", lambda *a, **k: None)
    session = {"_id": "test-session", "locked_entity": None}
    monkeypatch.setattr(chat_service.session_service, "recent_history", lambda *a, **k: [])
    ctx = {"session": session, "query": "什么是ETF？", "user_message_id": "u", "assistant_message_id": "a",
           "selected_doc_id": None}
    events = [chunk for chunk in chat_service.run_stream(ctx)]
    errors = [e for e in events if e.startswith("event: error")]
    assert errors, events[-3:]
    data = json.loads(errors[0].split("data: ", 1)[1])
    assert data["code"] == "LLM_ERROR" and "稍后重试" in data["message"]
