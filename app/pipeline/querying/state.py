"""查询流程的状态定义。

一次提问从 rewrite 节点开始，各节点依次填充下面的字段；流程结束时，
问答服务从最终状态中取出回答、引用、检索轨迹等保存到 MongoDB。
"""
from typing import Any, TypedDict


class QueryState(TypedDict, total=False):
    # ========== 输入（由问答服务提供） ==========
    query: str  # 用户原始问题
    history: list[dict[str, str]]  # 最近几轮对话 [{role: user/assistant, content}]，不含当前问题
    locked_entity: dict[str, Any] | None  # 会话中上一轮锁定的实体 {doc_ids, label}，用于理解"它/这个产品"
    selected_doc_id: str | None  # 用户在澄清卡片中选中的资料 ID（有值时直接锁定该资料）

    # ========== 问题理解（rewrite 节点） ==========
    rewritten_query: str  # 补全指代后的独立问题，后续检索都用它
    intent: str  # 问题类型：product / knowledge / market_policy / announcement / risk / process / advice / chitchat
    entity_mention: str  # 问题中提到的产品/公司/资料名称
    need_realtime: bool  # 是否涉及最新行情、净值等实时信息（为真时追加实时性提示）

    # ========== 实体确认（entity 节点） ==========
    entity: dict[str, Any] | None  # 本轮锁定的实体 {doc_ids, label}；检索时只在这些资料中查找
    clarify_candidates: list[dict[str, str]]  # 无法确定时给用户选择的候选资料

    # ========== 召回与排序（retrieve / rrf / rerank 节点） ==========
    dense_hits: list[dict[str, Any]]  # 稠密向量检索结果
    bm25_hits: list[dict[str, Any]]  # BM25 关键词检索结果
    web_hits: list[dict[str, Any]]  # 联网搜索结果
    fused: list[dict[str, Any]]  # RRF 融合后的候选
    reranked: list[dict[str, Any]]  # 重排后最终进入上下文的资料（带 score 相关度）

    # ========== 输出（answer / clarify / chitchat 节点） ==========
    outcome: str  # 结果类型：completed / no_answer / refused / clarify
    answer: str  # 最终回答全文（已经过禁用词替换）
    citations: list[dict[str, Any]]  # 引用来源，编号与回答中的 [n] 对应
    appendix: dict[str, str | None]  # 附加提示 {risk_notice, realtime_notice}，为 None 表示不显示
    timings: dict[str, int]  # 各阶段耗时（毫秒），保存到消息的 trace 中便于排查性能
