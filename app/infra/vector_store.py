"""Milvus 向量库：fin_chunks / fin_docs 两个集合（数据库设计文档 §2）。

- fin_chunks：每个文档切片一条，是问答检索的主体
- fin_docs：每份资料一条，存资料名、产品名、代码、机构等，用于"实体确认"
  （判断用户问的"华夏债券C"对应哪份资料）

每个集合同时有两种向量，支持混合检索：
- dense_vector：本地 bge-m3 生成的稠密向量，按语义相似度（COSINE）检索
- sparse_vector：Milvus 2.5 内置 BM25 Function 根据开启了中文分词的文本字段自动生成，
  按关键词匹配检索，适合产品代码、公司名这类需要精确命中的内容。写入时不需要提供该字段。
"""
import logging
from functools import lru_cache
from typing import Any

from pymilvus import DataType, Function, FunctionType, MilvusClient

from app.config import get_settings

logger = logging.getLogger(__name__)

# 中文分词器配置（Milvus 内置，基于 jieba），用于 BM25 全文检索
CHINESE_ANALYZER = {"type": "chinese"}

# 切片检索时需要返回的字段（不含向量本身，向量体积大且前端用不到）
CHUNK_OUTPUT_FIELDS = [
    "chunk_id", "doc_id", "chunk_index", "content", "content_type", "document_title", "product_name",
    "product_code", "institution_name", "risk_level", "industry", "market", "publish_date", "entry_name",
    "source_file", "source_path",
]
# 文档集合检索时需要返回的字段
DOC_OUTPUT_FIELDS = [
    "doc_id", "document_title", "product_name", "product_code", "institution_name", "content_type", "publish_date",
]


@lru_cache
def get_client() -> MilvusClient:
    """创建（并缓存）Milvus 客户端，请求超时 10 秒。"""
    return MilvusClient(uri=get_settings().milvus_uri, timeout=10)


def _bm25_function(input_field: str) -> Function:
    """定义 BM25 函数：Milvus 会对 input_field 的文本分词，自动生成稀疏向量写入 sparse_vector。"""
    return Function(
        name=f"{input_field}_bm25",
        function_type=FunctionType.BM25,
        input_field_names=[input_field],
        output_field_names=["sparse_vector"],
    )


def _index_params(client: MilvusClient, scalar_fields: list[str]):
    """构建索引参数：稠密向量用 HNSW、稀疏向量用 BM25 倒排，另外给常用过滤字段建标量倒排索引。"""
    params = client.prepare_index_params()
    # HNSW：图索引，召回率高、速度快；M=每个节点的邻居数，efConstruction=建索引时的搜索宽度
    params.add_index(field_name="dense_vector", index_type="HNSW", metric_type="COSINE",
                     params={"M": 16, "efConstruction": 200})
    params.add_index(field_name="sparse_vector", index_type="SPARSE_INVERTED_INDEX", metric_type="BM25")
    # 标量索引：加速 filter 表达式（如 doc_id in [...]、content_type == "faq"）
    for f in scalar_fields:
        params.add_index(field_name=f, index_type="INVERTED")
    return params


def _create_chunk_collection(client: MilvusClient, name: str, dim: int) -> None:
    """创建切片集合 fin_chunks。字段含义见数据库设计文档 §2.1。

    注意：Milvus 的 VARCHAR 长度按 UTF-8 字节计算，一个汉字占 3 字节。
    """
    schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field("chunk_id", DataType.VARCHAR, max_length=64, is_primary=True)  # {doc_id}_{序号}
    schema.add_field("doc_id", DataType.VARCHAR, max_length=32)
    schema.add_field("chunk_index", DataType.INT32)  # 切片在文档中的顺序
    schema.add_field("content", DataType.VARCHAR, max_length=16384)  # 切片正文，用于展示和放进上下文
    # search_text = 资料名 + 章节路径 + 正文，开启分词后作为 BM25 的输入，同时也是稠密向量的来源文本
    schema.add_field("search_text", DataType.VARCHAR, max_length=16384, enable_analyzer=True,
                     analyzer_params=CHINESE_ANALYZER)
    schema.add_field("dense_vector", DataType.FLOAT_VECTOR, dim=dim)
    schema.add_field("sparse_vector", DataType.SPARSE_FLOAT_VECTOR)  # 由 BM25 函数自动生成
    # 元数据字段：冗余到每个切片上，检索结果可以直接生成引用卡片，无需回查 MongoDB
    for f, length in [("content_type", 32), ("document_title", 512), ("product_name", 256), ("product_code", 64),
                      ("institution_name", 256), ("risk_level", 32), ("industry", 64), ("market", 64),
                      ("publish_date", 32), ("entry_name", 1024), ("source_file", 512), ("source_path", 1024)]:
        schema.add_field(f, DataType.VARCHAR, max_length=length)
    schema.add_function(_bm25_function("search_text"))
    client.create_collection(name, schema=schema,
                             index_params=_index_params(client, ["doc_id", "content_type"]))


def _create_doc_collection(client: MilvusClient, name: str, dim: int) -> None:
    """创建文档集合 fin_docs（每份资料一条），用于把用户提到的产品/公司匹配到具体资料。"""
    schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field("doc_id", DataType.VARCHAR, max_length=32, is_primary=True)
    for f, length in [("document_title", 512), ("product_name", 256), ("product_code", 64),
                      ("institution_name", 256), ("content_type", 32), ("publish_date", 32)]:
        schema.add_field(f, DataType.VARCHAR, max_length=length)
    # entity_text = 资料名 + 产品名 + 产品代码 + 机构名 + 简称，同时用于 BM25 和稠密向量
    schema.add_field("entity_text", DataType.VARCHAR, max_length=2048, enable_analyzer=True,
                     analyzer_params=CHINESE_ANALYZER)
    schema.add_field("dense_vector", DataType.FLOAT_VECTOR, dim=dim)
    schema.add_field("sparse_vector", DataType.SPARSE_FLOAT_VECTOR)
    schema.add_function(_bm25_function("entity_text"))
    client.create_collection(name, schema=schema, index_params=_index_params(client, ["content_type"]))


def ensure_collections() -> None:
    """确保两个集合存在并已加载到内存（Milvus 要求先 load 才能检索）。服务启动时调用。

    集合已存在时会检查向量维度是否与配置一致；不一致直接报错，不会自动删除重建，
    以免误删已导入的数据。
    """
    s, client = get_settings(), get_client()
    for name, creator in [(s.milvus_chunk_collection, _create_chunk_collection),
                          (s.milvus_doc_collection, _create_doc_collection)]:
        if client.has_collection(name):
            _check_dim(client, name, s.embedding_dim)
        else:
            creator(client, name, s.embedding_dim)
            logger.info("已创建 Milvus 集合 %s", name)
        client.load_collection(name)


def _check_dim(client: MilvusClient, name: str, dim: int) -> None:
    """检查已有集合的 dense_vector 维度与配置 EMBEDDING_DIM 是否一致（换了向量模型时会不一致）。"""
    desc = client.describe_collection(name)
    for field in desc["fields"]:
        if field["name"] == "dense_vector":
            actual = field["params"].get("dim")
            if int(actual) != dim:
                raise RuntimeError(f"集合 {name} 向量维度为 {actual}，与配置 EMBEDDING_DIM={dim} 不一致")


# ==================== 写入 ====================

def delete_doc(doc_id: str) -> None:
    """删除某份资料在两个集合中的全部数据。覆盖导入时先调用它清掉旧切片。"""
    s, client = get_settings(), get_client()
    expr = f'doc_id == "{doc_id}"'
    client.delete(s.milvus_chunk_collection, filter=expr)
    client.delete(s.milvus_doc_collection, filter=expr)


def insert_chunks(rows: list[dict[str, Any]], batch_size: int = 200) -> None:
    """批量写入切片。分批写入，避免单次请求过大（大报告可能有上百个切片）。"""
    s, client = get_settings(), get_client()
    for i in range(0, len(rows), batch_size):
        client.insert(s.milvus_chunk_collection, rows[i:i + batch_size])


def upsert_doc(row: dict[str, Any]) -> None:
    """写入或更新一条文档级记录（主键 doc_id 已存在时覆盖）。"""
    s = get_settings()
    get_client().upsert(s.milvus_doc_collection, [row])


# ==================== 检索 ====================

def _flatten(hits: list) -> list[dict[str, Any]]:
    """把 Milvus 返回的嵌套结果（每个查询一组命中）展开成字典列表，并把相似度放进 `_score` 字段。

    COSINE 检索时 `_score` 是余弦相似度（越大越相似）；BM25 检索时是 BM25 得分（无固定上限）。
    """
    results = []
    # 每次只查一个向量/文本，所以只取第一组结果 hits[0]
    for hit in hits[0] if hits else []:
        entity = dict(hit.get("entity", {}))
        entity["_score"] = float(hit["distance"])
        results.append(entity)
    return results


def dense_search(collection: str, vector: list[float], top_k: int, expr: str = "",
                 output_fields: list[str] | None = None) -> list[dict[str, Any]]:
    """稠密向量检索（语义相似）。

    Args:
        collection: 集合名
        vector: 查询向量（bge-m3 编码后的问题）
        top_k: 返回条数
        expr: 过滤表达式，如 'doc_id in ["xxx"]'；为空表示不过滤
        output_fields: 需要返回的字段，默认返回切片的全部元数据
    """
    hits = get_client().search(
        collection, data=[vector], anns_field="dense_vector", limit=top_k, filter=expr,
        # ef：HNSW 检索时的搜索宽度，必须 ≥ top_k，越大越准但越慢
        search_params={"metric_type": "COSINE", "params": {"ef": max(64, top_k * 2)}},
        output_fields=output_fields or CHUNK_OUTPUT_FIELDS,
    )
    return _flatten(hits)


def bm25_search(collection: str, text: str, top_k: int, expr: str = "",
                output_fields: list[str] | None = None) -> list[dict[str, Any]]:
    """BM25 关键词检索。直接传入原始文本，Milvus 会用同样的中文分词器分词后匹配。参数同 dense_search。"""
    hits = get_client().search(
        collection, data=[text], anns_field="sparse_vector", limit=top_k, filter=expr,
        search_params={"metric_type": "BM25"},
        output_fields=output_fields or CHUNK_OUTPUT_FIELDS,
    )
    return _flatten(hits)


def ping() -> bool:
    """健康检查：连不上会抛异常。"""
    get_client().list_collections()
    return True
