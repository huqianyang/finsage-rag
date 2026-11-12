"""查询流程图：

rewrite ─┬─ chitchat ─────────────────────────────────────→ END
         └─ entity ─┬─ clarify ───────────────────────────→ END
                    └─ retrieve(稠密/BM25/联网 并行) → rrf → rerank → answer → END
"""
from functools import lru_cache

from langgraph.graph import END, START, StateGraph

from app.pipeline.querying import nodes as n
from app.pipeline.querying.state import QueryState


@lru_cache
def build_query_graph():
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
    g.add_conditional_edges("rewrite", n.route_after_rewrite, {"chitchat": "chitchat", "entity": "entity"})
    g.add_conditional_edges("entity", n.route_after_entity, {"clarify": "clarify", "retrieve": "retrieve"})
    g.add_edge("retrieve", "rrf")
    g.add_edge("rrf", "rerank")
    g.add_edge("rerank", "answer")
    g.add_edge("answer", END)
    g.add_edge("clarify", END)
    g.add_edge("chitchat", END)
    return g.compile()
