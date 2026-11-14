"""查询流程节点（架构设计文档 §5）。

节点之间通过 QueryState 传递数据；同时通过 LangGraph 的 custom stream writer
向外推送实时事件，问答服务把它们转成 SSE 发给浏览器：
  ("status", {...})   阶段提示，如"正在检索资料…"
  ("clarify", {...})  需要用户在候选资料中选择
  ("delta", {...})    回答的一段新文字
  ("replace", {...})  合规替换（禁用表达被替换）
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
    """向流程外部推送一个事件（只在 graph.stream(stream_mode 含 "custom") 运行时有效）。"""
    get_stream_writer()((event, data))


def _ms(t0: float) -> int:
    """从 t0（time.perf_counter 的值）到现在经过的毫秒数。"""
    return int((time.perf_counter() - t0) * 1000)


def _history_text(history: list[dict[str, str]], limit_chars: int = 300) -> str:
    """把对话历史拼成"用户：…/助手：…"的文本放进提示词。

    每条只取前 limit_chars 个字：历史回答往往很长，全部放进去会浪费 token 并拖慢响应，
    而理解指代只需要知道之前聊的是什么。
    """
    if not history:
        return "（无）"
    lines = []
    for m in history:
        role = "用户" if m["role"] == "user" else "助手"
        lines.append(f"{role}：{m['content'][:limit_chars]}")
    return "\n".join(lines)


# ==================== 1. 问题理解 ====================
def rewrite_node(state: QueryState) -> dict:
    """调用大模型理解问题：补全指代、判断问题类型、提取实体名称、判断是否涉及实时信息。

    例：上一轮在聊华夏债券C，本轮问"它赎回有没有限制？"
        → rewritten_query = "华夏债券C（001003）赎回有没有限制？"，intent = product
    大模型调用失败时降级：用原问题检索、类型按 knowledge 处理，问答不中断。
    """
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
    # 模型给出的类型不在预定义列表中时，按"金融知识"处理
    intent = data.get("intent") if data.get("intent") in P.INTENTS else "knowledge"
    return {
        "rewritten_query": (data.get("rewritten_query") or state["query"]).strip(),
        "intent": intent,
        "entity_mention": (data.get("entity_mention") or "").strip(),
        "need_realtime": bool(data.get("need_realtime")),
        "timings": {"rewrite_ms": _ms(t0)},
    }


def route_after_rewrite(state: QueryState) -> str:
    """路由：闲聊直接回应，其余进入实体确认。"""
    return "chitchat" if state["intent"] == "chitchat" else "entity"


# ==================== 2. 实体确认 ====================
def entity_node(state: QueryState) -> dict:
    """实体确认节点：调用 _resolve_entity 并记录耗时。"""
    t0 = time.perf_counter()
    result = _resolve_entity(state)
    result["timings"] = {**state.get("timings", {}), "entity_ms": _ms(t0)}
    return result


def _resolve_entity(state: QueryState) -> dict:
    """把问题中提到的产品/公司匹配到具体资料。

    结果有三种：
    - 锁定 {"entity": {doc_ids, label}}：后续检索只在这份资料里找，避免把别的产品信息混进来
    - 澄清 {"clarify_candidates": [...]}：有多个可能的资料且无法区分，让用户选
    - 不锁定 {"entity": None}：问题不针对具体资料（如"什么是ETF"），在全库检索

    匹配方法：在 fin_docs 中用稠密向量和 BM25 各召回 5 条候选，再用重排模型打分：
    - 最高分 ≥ 高置信阈值，且比第二名高出足够多 → 锁定
    - 有 2 个及以上候选达到中等置信 → 澄清
    - 只有 1 个达到中等置信 → 锁定
    - 都不够 → 不锁定
    """
    s = get_settings()
    # 用户在澄清卡片中已经选了资料：直接锁定，不再匹配
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
    # 两路召回合并去重（同一资料只保留第一次出现的记录）
    cands: dict[str, dict] = {}
    for hit in vector_store.dense_search(col, vec, 5, output_fields=vector_store.DOC_OUTPUT_FIELDS + ["entity_text"]) + \
            vector_store.bm25_search(col, mention, 5, output_fields=vector_store.DOC_OUTPUT_FIELDS + ["entity_text"]):
        cands.setdefault(hit["doc_id"], hit)
    if not cands:
        return {"entity": None}

    docs = list(cands.values())
    try:
        scores = local_models.rerank(mention, [d["entity_text"] or d["document_title"] for d in docs])
    except Exception as e:  # 降级：无法判断置信度时不锁定实体，按全库检索
        logger.warning("实体确认重排失败，不锁定实体：%s", e)
        return {"entity": None}
    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    top, top_score = ranked[0]
    second = ranked[1][1] if len(ranked) > 1 else 0.0
    logger.info("实体匹配 %s → %s", mention, [(_doc_label(d), round(sc, 3)) for d, sc in ranked[:4]])

    # 高置信且领先明显：直接锁定第一名
    if top_score >= s.entity_high_confidence and top_score - second >= s.entity_score_gap:
        return {"entity": {"doc_ids": [top["doc_id"]], "label": _doc_label(top)}}
    # 中等置信的候选（最多 entity_max_options 个）
    mids = [(d, sc) for d, sc in ranked if sc >= s.entity_mid_confidence][: s.entity_max_options]
    if len(mids) >= 2:
        return {"clarify_candidates": [{"doc_id": d["doc_id"], "title": _doc_label(d),
                                        "content_type_label": label_of(d["content_type"]),
                                        "publish_date": d.get("publish_date", "")} for d, _ in mids]}
    if len(mids) == 1:
        return {"entity": {"doc_ids": [top["doc_id"]], "label": _doc_label(top)}}
    return {"entity": None}


def _doc_label(doc: dict) -> str:
    """资料的显示名称：优先资料名，其次产品名，最后用 ID 兜底。"""
    return doc.get("document_title") or doc.get("product_name") or doc.get("doc_id")


def route_after_entity(state: QueryState) -> str:
    """路由：有候选需要澄清则走 clarify，否则进入检索。"""
    return "clarify" if state.get("clarify_candidates") else "retrieve"


def clarify_node(state: QueryState) -> dict:
    """推送澄清事件，本轮到此结束；用户点选候选后，前端带着 selected_doc_id 重新发起本问题。"""
    text = "知识库中有以下相关资料，你想了解的是哪一个？"
    _emit("clarify", {"text": text, "candidates": state["clarify_candidates"]})
    return {"outcome": "clarify", "answer": text, "citations": [], "appendix": {}}


# ==================== 3. 召回（三路并行） ====================
def retrieve_node(state: QueryState) -> dict:
    """稠密 / BM25 / 联网 三路并行召回。

    - 稠密向量：按语义相似检索，能匹配到换了说法的内容
    - BM25：按关键词检索，产品代码、公司名等能精确命中
    - 联网搜索：补充知识库没有的内容（结果后续会标注为网络来源）
    三路互不依赖，用线程池同时执行，总耗时取决于最慢的一路（通常是联网搜索）。
    """
    _emit("status", {"stage": "retrieve", "text": "正在检索资料…"})
    s, t0 = get_settings(), time.perf_counter()
    query = state["rewritten_query"]
    entity = state.get("entity")
    # 锁定了实体时，知识库检索只在该资料的切片中进行
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


# ==================== 4. RRF 融合 ====================
def rrf_node(state: QueryState) -> dict:
    """用 RRF（Reciprocal Rank Fusion，倒数排名融合）合并三路结果。

    三路检索的分数量纲不同（余弦相似度、BM25 分、网页无分数），无法直接比较，
    RRF 只看排名：某条结果在每一路中排第 rank 名，就得 1/(k+rank) 分，各路得分相加。
    被多路同时检索到、且排名靠前的结果总分最高。k 是平滑常数（默认 60）。
    """
    s = get_settings()
    scores: dict[str, float] = {}  # 唯一键 → RRF 总分
    items: dict[str, dict] = {}  # 唯一键 → 结果内容
    for hits, source in [(state.get("dense_hits", []), "kb"), (state.get("bm25_hits", []), "kb"),
                         (state.get("web_hits", []), "web")]:
        for rank, hit in enumerate(hits, start=1):
            # 知识库结果以切片 ID 去重，网络结果以 URL（没有则标题）去重
            key = hit["chunk_id"] if source == "kb" else f"web:{hit['url'] or hit['title']}"
            scores[key] = scores.get(key, 0.0) + 1.0 / (s.rrf_k + rank)
            items.setdefault(key, {**hit, "source_type": source, "key": key})
    fused = sorted(items.values(), key=lambda x: scores[x["key"]], reverse=True)[: s.rrf_top_n]
    for f in fused:
        f["rrf_score"] = round(scores[f["key"]], 5)
    return {"fused": fused}


# ==================== 5. 重排 ====================
def _passage(item: dict) -> str:
    """送入重排模型的文本：网络结果用标题+摘要，知识库切片用资料名+章节+正文。"""
    if item["source_type"] == "web":
        return f"{item['title']}\n{item['snippet']}"
    return f"{item['document_title']}\n{item['entry_name']}\n{item['content']}"


def rerank_node(state: QueryState) -> dict:
    """用本地重排模型给 RRF 融合后的候选逐条打相关度分（0~1），取最相关的前几条。

    重排模型不可用时降级：保留 RRF 顺序，score 置为 None（回答节点据此跳过相关度判断）。
    """
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
    """保持重排顺序取前 top_n 条，但网络来源最多 web_max 条，避免网页挤掉知识库资料。

    联调时发现：网页标题常直接包含产品名，重排分数会高于知识库切片，
    不加限制时前几名可能全是网页，违背"基于知识库作答"的定位。
    """
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


# ==================== 6. 生成 ====================
def _select_context(state: QueryState) -> tuple[list[dict], str]:
    """挑选进入大模型上下文的资料，并判断回答类型。

    Returns:
        (资料列表, 类型)，类型为：
        - normal：有相关的知识库资料（可能夹带网络资料），正常回答
        - web_only：知识库没有相关资料，只有网络资料，回答需以"以下内容来自网络"开头
        - none：都没有相关资料，回答固定兜底话术（不调用大模型，避免编造）
    "相关"的标准是重排分数 ≥ NO_ANSWER_SCORE_THRESHOLD。
    """
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
    """把资料拼成带编号的参考资料文本：[1]（资料：…；类型：…；章节：…）正文…

    编号与最终引用卡片一一对应，模型在回答中用 [n] 标注引用。
    知识库资料附带资料名、产品代码、发布时间等元数据，方便模型准确引用；网络资料标明"网络来源"。
    """
    blocks = []
    for i, x in enumerate(items, start=1):
        if x["source_type"] == "web":
            blocks.append(f"[{i}]（网络来源）{x['title']}\n{x['snippet']}")
        else:
            # filter(None, ...) 去掉值为空的元数据项
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
    """生成第 i 条引用来源（前端渲染为引用卡片）。由系统根据实际使用的资料生成，不由模型编写。

    只返回给用户看的字段，不暴露 chunk_id 等内部 ID；doc_id 保留，"查看原文"接口要用它。
    具体命中了哪些切片记录在消息的 trace 中（只存数据库，不返回前端），排查问题时去那里查。
    """
    score = round(x["score"], 4) if x.get("score") is not None else None
    if x["source_type"] == "web":
        return {"index": i, "source_type": "web", "title": x["title"], "url": x["url"],
                "snippet": x["snippet"][:300], "score": score}
    return {"index": i, "source_type": "kb", "doc_id": x["doc_id"],
            "document_title": x["document_title"], "content_type": x["content_type"],
            "content_type_label": label_of(x["content_type"]), "product_name": x.get("product_name", ""),
            "product_code": x.get("product_code", ""), "publish_date": x.get("publish_date", ""),
            "source_file": x.get("source_file", ""), "entry_name": x.get("entry_name", ""),
            "snippet": x["content"][:300], "score": score}


def _stream_answer(messages: list[dict]) -> str:
    """流式调用大模型，边生成边推送 delta 事件，并实时检测禁用表达。

    每收到一段文字：先推送原文（delta），如果检测到禁用表达，再推送替换事件（replace），
    前端收到后修正已显示的文字。返回替换后的最终全文。
    """
    _emit("status", {"stage": "generate", "text": "正在整理答案…"})
    guard = rules.StreamGuard()
    for delta in llm.chat_stream(messages):
        reps = guard.feed(delta)
        _emit("delta", {"text": delta})
        for r in reps:
            _emit("replace", {"start": r.start, "end": r.end, "text": r.text})
    return guard.text


def answer_node(state: QueryState) -> dict:
    """生成最终回答。

    - 无相关资料：直接输出固定兜底话术（投资建议类问题前面加拒答话术），不调用大模型；
    - 有资料：按问题类型选择回答指引，把编号后的资料、对话历史、问题交给大模型流式生成。
    投资建议类问题的 outcome 为 refused（回答开头有拒答话术，提示词中已要求）。
    """
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
    """计算回答末尾的附加提示。

    - 风险提示：基于资料作答（或属于投资建议类问题）且命中风险提示规则时追加；
      无资料的兜底回答本身不涉及产品内容，不追加。
    - 实时性提示：问题涉及最新行情、净值等时追加。
    """
    risk = rules.RISK_NOTICE if (kb_used or state["intent"] == "advice") and \
        rules.need_risk_notice(state["intent"], answer) else None
    realtime = rules.REALTIME_NOTICE if state.get("need_realtime") else None
    return {"risk_notice": risk, "realtime_notice": realtime}


def chitchat_node(state: QueryState) -> dict:
    """闲聊：用原问题让大模型简短回应并介绍可查询的范围，不检索、不附引用和提示。"""
    messages = [{"role": "system", "content": P.CHITCHAT_SYSTEM}, {"role": "user", "content": state["query"]}]
    answer = _stream_answer(messages)
    return {"outcome": "completed", "answer": answer, "citations": [],
            "appendix": {"risk_notice": None, "realtime_notice": None}}
