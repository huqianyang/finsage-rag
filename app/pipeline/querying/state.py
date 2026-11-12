from typing import Any, TypedDict


class QueryState(TypedDict, total=False):
    # 输入
    query: str
    history: list[dict[str, str]]  # [{role, content}]，不含当前问题
    locked_entity: dict[str, Any] | None  # {doc_ids, label}
    selected_doc_id: str | None
    # 理解
    rewritten_query: str
    intent: str
    entity_mention: str
    need_realtime: bool
    # 实体
    entity: dict[str, Any] | None  # 本轮锁定 {doc_ids, label}
    clarify_candidates: list[dict[str, str]]
    # 召回
    dense_hits: list[dict[str, Any]]
    bm25_hits: list[dict[str, Any]]
    web_hits: list[dict[str, Any]]
    fused: list[dict[str, Any]]
    reranked: list[dict[str, Any]]
    # 输出
    outcome: str  # completed / no_answer / refused / clarify
    answer: str
    citations: list[dict[str, Any]]
    appendix: dict[str, str | None]
    timings: dict[str, int]
