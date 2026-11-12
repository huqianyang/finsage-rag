"""查询流程节点（架构设计 §5）。

节点通过 LangGraph 的 custom stream writer 向外推送事件：
  ("status", {...}) / ("clarify", {...}) / ("delta", {...}) / ("replace", {...})
"""
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from langgraph.config import get_stream_writer

from app.common.content_types import label_of
from app.compliance import rules
from app.config import get_settings
from app.infra import llm, local_models, vector_store, web_search
from app.pipeline.querying.state import QueryState
from app.prompts import querying as P

logger = logging.getLogger(__name__)


def _emit(event: str, data: Any) -> None:
    get_stream_writer()((event, data))


def _ms(t0: float) -> int:
    return int((time.perf_counter() - t0) * 1000)


def _history_text(history: list[dict[str, str]], limit_chars: int = 300) -> str:
    if not history:
        return "（无）"
    lines = []
    for m in history:
        role = "用户" if m["role"] == "user" else "助手"
        lines.append(f"{role}：{m['content'][:limit_chars]}")
    return "\n".join(lines)


# ---------- 1. 问题理解 ----------
def rewrite_node(state: QueryState) -> dict:
    _emit("status", {"stage": "rewrite", "text": "正在理解问题…"})
    t0 = time.perf_counter()
    locked = (state.get("locked_entity") or {}).get("label") or "（无）"
    try:
        data = llm.chat_json([
            {"role": "system", "content": P.REWRITE_SYSTEM},
            {"role": "user", "content": P.REWRITE_USER.format(
                locked=locked, history=_history_text(state.get("history", [])), query=state["query"])},
        ], temperature=0, max_tokens=400)
    except Exception as e:  # 理解失败时降级：原问题直接检索
        logger.warning("问题改写失败，使用原问题：%s", e)
        data = {}
    intent = data.get("intent") if data.get("intent") in P.INTENTS else "knowledge"
    return {
        "rewritten_query": (data.get("rewritten_query") or state["query"]).strip(),
        "intent": intent,
        "entity_mention": (data.get("entity_mention") or "").strip(),
        "need_realtime": bool(data.get("need_realtime")),
        "timings": {"rewrite_ms": _ms(t0)},
    }


def route_after_rewrite(state: QueryState) -> str:
    return "chitchat" if state["intent"] == "chitchat" else "entity"


# ---------- 2. 实体确认 ----------
def entity_node(state: QueryState) -> dict:
    t0 = time.perf_counter()
    result = _resolve_entity(state)
    result["timings"] = {**state.get("timings", {}), "entity_ms": _ms(t0)}
    return result


def _resolve_entity(state: QueryState) -> dict:
    s = get_settings()
    selected = state.get("selected_doc_id")
    if selected:
        rows = vector_store.get_client().get(s.milvus_doc_collection, ids=[selected],
                                             output_fields=vector_store.DOC_OUTPUT_FIELDS)
        if rows:
            return {"entity": {"doc_ids": [selected], "label": _doc_label(rows[0])}}

    mention = state.get("entity_mention")
    if not mention:
        return {"entity": None}

    _emit("status", {"stage": "entity", "text": "正在匹配相关资料…"})
    col = s.milvus_doc_collection
    vec = local_models.embed([mention])[0]
    cands: dict[str, dict] = {}
    for hit in vector_store.dense_search(col, vec, 5, output_fields=vector_store.DOC_OUTPUT_FIELDS + ["entity_text"]) + \
            vector_store.bm25_search(col, mention, 5, output_fields=vector_store.DOC_OUTPUT_FIELDS + ["entity_text"]):
        cands.setdefault(hit["doc_id"], hit)
    if not cands:
        return {"entity": None}

    docs = list(cands.values())
    scores = local_models.rerank(mention, [d["entity_text"] or d["document_title"] for d in docs])
    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    top, top_score = ranked[0]
    second = ranked[1][1] if len(ranked) > 1 else 0.0
    logger.info("实体匹配 %s → %s", mention, [(_doc_label(d), round(sc, 3)) for d, sc in ranked[:4]])

    if top_score >= s.entity_high_confidence and top_score - second >= s.entity_score_gap:
        return {"entity": {"doc_ids": [top["doc_id"]], "label": _doc_label(top)}}
    mids = [(d, sc) for d, sc in ranked if sc >= s.entity_mid_confidence][: s.entity_max_options]
    if len(mids) >= 2:
        return {"clarify_candidates": [{"doc_id": d["doc_id"], "title": _doc_label(d),
                                        "content_type_label": label_of(d["content_type"]),
                                        "publish_date": d.get("publish_date", "")} for d, _ in mids]}
    if len(mids) == 1:
        return {"entity": {"doc_ids": [top["doc_id"]], "label": _doc_label(top)}}
    return {"entity": None}


def _doc_label(doc: dict) -> str:
    return doc.get("document_title") or doc.get("product_name") or doc.get("doc_id")


def route_after_entity(state: QueryState) -> str:
    return "clarify" if state.get("clarify_candidates") else "retrieve"


def clarify_node(state: QueryState) -> dict:
    text = "知识库中有以下相关资料，你想了解的是哪一个？"
    _emit("clarify", {"text": text, "candidates": state["clarify_candidates"]})
    return {"outcome": "clarify", "answer": text, "citations": [], "appendix": {}}


# ---------- 3. 召回（三路并行） ----------
def retrieve_node(state: QueryState) -> dict:
    """稠密 / BM25 / 联网 三路并行召回。"""
    _emit("status", {"stage": "retrieve", "text": "正在检索资料…"})
    s, t0 = get_settings(), time.perf_counter()
    query = state["rewritten_query"]
    entity = state.get("entity")
    expr = ""
    if entity and entity.get("doc_ids"):
        ids = ", ".join(f'"{d}"' for d in entity["doc_ids"])
        expr = f"doc_id in [{ids}]"
    col = s.milvus_chunk_collection

    def dense():
        return vector_store.dense_search(col, local_models.embed([query])[0], s.dense_top_k, expr)

    def bm25():
        return vector_store.bm25_search(col, query, s.bm25_top_k, expr)

    with ThreadPoolExecutor(max_workers=3) as pool:
        f_dense, f_bm25, f_web = pool.submit(dense), pool.submit(bm25), pool.submit(web_search.search, query)
        dense_hits, bm25_hits = f_dense.result(), f_bm25.result()
        web_hits = f_web.result()  # 内部已降级，不会抛错
    timings = {**state.get("timings", {}), "retrieve_ms": _ms(t0)}
    return {"dense_hits": dense_hits, "bm25_hits": bm25_hits, "web_hits": web_hits, "timings": timings}


# ---------- 4. RRF 融合 ----------
def rrf_node(state: QueryState) -> dict:
    s = get_settings()
    scores: dict[str, float] = {}
    items: dict[str, dict] = {}
    for hits, source in [(state.get("dense_hits", []), "kb"), (state.get("bm25_hits", []), "kb"),
                         (state.get("web_hits", []), "web")]:
        for rank, hit in enumerate(hits, start=1):
            key = hit["chunk_id"] if source == "kb" else f"web:{hit['url'] or hit['title']}"
            scores[key] = scores.get(key, 0.0) + 1.0 / (s.rrf_k + rank)
            items.setdefault(key, {**hit, "source_type": source, "key": key})
    fused = sorted(items.values(), key=lambda x: scores[x["key"]], reverse=True)[: s.rrf_top_n]
    for f in fused:
        f["rrf_score"] = round(scores[f["key"]], 5)
    return {"fused": fused}


# ---------- 5. 重排 ----------
def _passage(item: dict) -> str:
    if item["source_type"] == "web":
        return f"{item['title']}\n{item['snippet']}"
    return f"{item['document_title']}\n{item['entry_name']}\n{item['content']}"


def rerank_node(state: QueryState) -> dict:
    s, t0 = get_settings(), time.perf_counter()
    fused = state.get("fused", [])
    if not fused:
        return {"reranked": []}
    _emit("status", {"stage": "rerank", "text": "正在筛选最相关的资料…"})
    try:
        scores = local_models.rerank(state["rewritten_query"], [_passage(x) for x in fused])
        ranked = [{**x, "score": sc} for x, sc in zip(fused, scores)]
        ranked.sort(key=lambda x: x["score"], reverse=True)
    except Exception as e:  # 降级：按 RRF 顺序
        logger.warning("重排失败，使用 RRF 顺序：%s", e)
        ranked = [{**x, "score": None} for x in fused]
    timings = {**state.get("timings", {}), "rerank_ms": _ms(t0)}
    return {"reranked": _cap_web(ranked, s.rerank_top_n, s.web_max_in_context), "timings": timings}


def _cap_web(ranked: list[dict], top_n: int, web_max: int) -> list[dict]:
    """保持重排顺序取前 top_n 条，但网络来源最多 web_max 条，避免网页挤掉知识库资料。"""
    out, web = [], 0
    for x in ranked:
        if x["source_type"] == "web":
            if web >= web_max:
                continue
            web += 1
        out.append(x)
        if len(out) >= top_n:
            break
    return out


# ---------- 6. 生成 ----------
def _select_context(state: QueryState) -> tuple[list[dict], str]:
    """返回（进入上下文的资料, 结果类型 normal / web_only / none）。"""
    th = get_settings().no_answer_score_threshold
    ranked = state.get("reranked", [])
    if ranked and ranked[0]["score"] is None:  # 重排降级：无法判断相关度，取前几条
        return ranked, "normal" if any(x["source_type"] == "kb" for x in ranked) else "web_only"
    kb = [x for x in ranked if x["source_type"] == "kb" and x["score"] >= th]
    web = [x for x in ranked if x["source_type"] == "web" and x["score"] >= th]
    if kb:
        return [x for x in ranked if x["score"] >= th], "normal"
    if web:
        return web, "web_only"
    return [], "none"


def _context_text(items: list[dict]) -> str:
    blocks = []
    for i, x in enumerate(items, start=1):
        if x["source_type"] == "web":
            blocks.append(f"[{i}]（网络来源）{x['title']}\n{x['snippet']}")
        else:
            meta = "；".join(filter(None, [
                f"资料：{x['document_title']}", f"类型：{label_of(x['content_type'])}",
                x.get("product_name") and f"产品：{x['product_name']}",
                x.get("product_code") and f"代码：{x['product_code']}",
                x.get("publish_date") and f"发布时间：{x['publish_date']}",
                x.get("entry_name") and f"章节：{x['entry_name']}",
            ]))
            blocks.append(f"[{i}]（{meta}）\n{x['content']}")
    return "\n\n".join(blocks)


def _citation(i: int, x: dict) -> dict:
    score = round(x["score"], 4) if x.get("score") is not None else None
    if x["source_type"] == "web":
        return {"index": i, "source_type": "web", "title": x["title"], "url": x["url"],
                "snippet": x["snippet"][:300], "score": score}
    return {"index": i, "source_type": "kb", "chunk_id": x["chunk_id"], "doc_id": x["doc_id"],
            "document_title": x["document_title"], "content_type": x["content_type"],
            "content_type_label": label_of(x["content_type"]), "product_name": x.get("product_name", ""),
            "product_code": x.get("product_code", ""), "publish_date": x.get("publish_date", ""),
            "source_file": x.get("source_file", ""), "entry_name": x.get("entry_name", ""),
            "snippet": x["content"][:300], "score": score}


def _stream_answer(messages: list[dict]) -> str:
    _emit("status", {"stage": "generate", "text": "正在整理答案…"})
    guard = rules.StreamGuard()
    for delta in llm.chat_stream(messages):
        reps = guard.feed(delta)
        _emit("delta", {"text": delta})
        for r in reps:
            _emit("replace", {"start": r.start, "end": r.end, "text": r.text})
    return guard.text


def answer_node(state: QueryState) -> dict:
    t0 = time.perf_counter()
    items, kind = _select_context(state)
    intent = state["intent"]
    timings = dict(state.get("timings", {}))

    if kind == "none":
        text = rules.NO_ANSWER_TEXT
        if intent == "advice":
            text = rules.REFUSE_PREFIX + "\n\n" + text
        _emit("delta", {"text": text})
        return {"outcome": "refused" if intent == "advice" else "no_answer", "answer": text, "citations": [],
                "appendix": _appendix(state, text, kb_used=False), "timings": timings}

    messages = [
        {"role": "system", "content": P.ANSWER_SYSTEM},
        {"role": "user", "content": P.ANSWER_USER.format(
            guide=P.INTENT_GUIDE.get(intent, ""), web_only_note=P.WEB_ONLY_NOTE if kind == "web_only" else "",
            context=_context_text(items), history=_history_text(state.get("history", [])),
            query=state["rewritten_query"])},
    ]
    answer = _stream_answer(messages)
    timings["generate_ms"] = _ms(t0)
    return {
        "outcome": "refused" if intent == "advice" else "completed",
        "answer": answer,
        "citations": [_citation(i, x) for i, x in enumerate(items, start=1)],
        "appendix": _appendix(state, answer, kb_used=True),
        "timings": timings,
    }


def _appendix(state: QueryState, answer: str, kb_used: bool) -> dict:
    risk = rules.RISK_NOTICE if (kb_used or state["intent"] == "advice") and \
        rules.need_risk_notice(state["intent"], answer) else None
    realtime = rules.REALTIME_NOTICE if state.get("need_realtime") else None
    return {"risk_notice": risk, "realtime_notice": realtime}


def chitchat_node(state: QueryState) -> dict:
    messages = [{"role": "system", "content": P.CHITCHAT_SYSTEM}, {"role": "user", "content": state["query"]}]
    answer = _stream_answer(messages)
    return {"outcome": "completed", "answer": answer, "citations": [],
            "appendix": {"risk_notice": None, "realtime_notice": None}}
