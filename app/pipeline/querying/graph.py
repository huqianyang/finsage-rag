"""查询流程图（LangGraph）：

rewrite ─┬─ chitchat ─────────────────────────────────────→ END
         └─ entity ─┬─ clarify ───────────────────────────→ END
                    └─ retrieve(稠密/BM25/联网 并行) → rrf → rerank → answer → END

- rewrite：理解问题（补全指代、判断类型、提取实体名称）
- chitchat：闲聊直接简短回应，不检索
- entity：把提到的产品/公司匹配到具体资料；不确定时走 clarify 让用户选择
- retrieve → rrf → rerank：三路召回、融合、重排
- answer：判断是否有资料可答，流式生成回答并做合规处理
"""
from functools import lru_cache

from langgraph.graph import END, START, StateGraph

from app.pipeline.querying import nodes as n
from app.pipeline.querying.state import QueryState


@lru_cache
def build_query_graph():
    """构建并编译查询流程图（只编译一次，多个请求并发复用）。"""
    g = StateGraph(QueryState)
    g.add_node("rewrite", n.rewrite_node)
    g.add_node("chitchat", n.chitchat_node)
    g.add_node("entity", n.entity_node)
    g.add_node("clarify", n.clarify_node)
    g.add_node("retrieve", n.retrieve_node)
    g.add_node("rrf", n.rrf_node)
    g.add_node("rerank", n.rerank_node)
    g.add_node("answer", n.answer_node)

    g.add_edge(START, "rewrite")
    # 条件边：路由函数返回的字符串决定下一个节点
    g.add_conditional_edges("rewrite", n.route_after_rewrite, {"chitchat": "chitchat", "entity": "entity"})
    g.add_conditional_edges("entity", n.route_after_entity, {"clarify": "clarify", "retrieve": "retrieve"})
    g.add_edge("retrieve", "rrf")
    g.add_edge("rrf", "rerank")
    g.add_edge("rerank", "answer")
    # 三个出口
    g.add_edge("answer", END)
    g.add_edge("clarify", END)
    g.add_edge("chitchat", END)
    return g.compile()
