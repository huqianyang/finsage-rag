"""导入流程图（LangGraph）：parse → image → split → metadata → embed → index。

这是一条没有分支的线性流程，任一节点抛异常都会终止整个流程，
由导入服务（import_service._run）捕获并把任务标记为失败。
"""
from functools import lru_cache

from langgraph.graph import END, START, StateGraph

from app.pipeline.importing import nodes
from app.pipeline.importing.state import ImportState


@lru_cache
def build_import_graph():
    """构建并编译导入流程图。编译后的图可被多个线程并发复用，lru_cache 保证只编译一次。"""
    g = StateGraph(ImportState)
    # 注册节点：名称 → 处理函数
    g.add_node("parse", nodes.parse_node)  # 文件 → Markdown
    g.add_node("image", nodes.image_node)  # 图片上传 MinIO + 生成摘要
    g.add_node("split", nodes.split_node)  # Markdown → 切片
    g.add_node("metadata", nodes.metadata_node)  # 大模型抽取元数据
    g.add_node("embed", nodes.embed_node)  # 切片 → 向量
    g.add_node("index", nodes.index_node)  # 写入 Milvus
    # 连线：按顺序依次执行
    g.add_edge(START, "parse")
    g.add_edge("parse", "image")
    g.add_edge("image", "split")
    g.add_edge("split", "metadata")
    g.add_edge("metadata", "embed")
    g.add_edge("embed", "index")
    g.add_edge("index", END)
    return g.compile()
