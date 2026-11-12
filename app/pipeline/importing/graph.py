"""导入流程图：parse → image → split → metadata → embed → index。"""
from functools import lru_cache

from langgraph.graph import END, START, StateGraph

from app.pipeline.importing import nodes
from app.pipeline.importing.state import ImportState


@lru_cache
def build_import_graph():
    g = StateGraph(ImportState)
    g.add_node("parse", nodes.parse_node)
    g.add_node("image", nodes.image_node)
    g.add_node("split", nodes.split_node)
    g.add_node("metadata", nodes.metadata_node)
    g.add_node("embed", nodes.embed_node)
    g.add_node("index", nodes.index_node)
    g.add_edge(START, "parse")
    g.add_edge("parse", "image")
    g.add_edge("image", "split")
    g.add_edge("split", "metadata")
    g.add_edge("metadata", "embed")
    g.add_edge("embed", "index")
    g.add_edge("index", END)
    return g.compile()
