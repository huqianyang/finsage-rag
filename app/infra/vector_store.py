"""Milvus 向量库：fin_chunks / fin_docs 两个集合（见 数据库设计 §2）。

每个集合同时有：
- dense_vector：本地 bge-m3 生成的稠密向量（COSINE）
- sparse_vector：Milvus 内置 BM25 Function 根据 analyzer 字段自动生成（中文分词）
"""
import logging
from functools import lru_cache
from typing import Any

from pymilvus import DataType, Function, FunctionType, MilvusClient

from app.config import get_settings

logger = logging.getLogger(__name__)

CHINESE_ANALYZER = {"type": "chinese"}

# 切片集合中除向量外需要返回的字段
CHUNK_OUTPUT_FIELDS = [
    "chunk_id", "doc_id", "chunk_index", "content", "content_type", "document_title", "product_name",
    "product_code", "institution_name", "risk_level", "industry", "market", "publish_date", "entry_name",
    "source_file", "source_path",
]
DOC_OUTPUT_FIELDS = [
    "doc_id", "document_title", "product_name", "product_code", "institution_name", "content_type", "publish_date",
]


@lru_cache
def get_client() -> MilvusClient:
    return MilvusClient(uri=get_settings().milvus_uri, timeout=10)


def _bm25_function(input_field: str) -> Function:
    return Function(
        name=f"{input_field}_bm25",
        function_type=FunctionType.BM25,
        input_field_names=[input_field],
        output_field_names=["sparse_vector"],
    )


def _index_params(client: MilvusClient, scalar_fields: list[str]):
    params = client.prepare_index_params()
    params.add_index(field_name="dense_vector", index_type="HNSW", metric_type="COSINE",
                     params={"M": 16, "efConstruction": 200})
    params.add_index(field_name="sparse_vector", index_type="SPARSE_INVERTED_INDEX", metric_type="BM25")
    for f in scalar_fields:
        params.add_index(field_name=f, index_type="INVERTED")
    return params


def _create_chunk_collection(client: MilvusClient, name: str, dim: int) -> None:
    schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field("chunk_id", DataType.VARCHAR, max_length=64, is_primary=True)
    schema.add_field("doc_id", DataType.VARCHAR, max_length=32)
    schema.add_field("chunk_index", DataType.INT32)
    schema.add_field("content", DataType.VARCHAR, max_length=16384)
    schema.add_field("search_text", DataType.VARCHAR, max_length=16384, enable_analyzer=True,
                     analyzer_params=CHINESE_ANALYZER)
    schema.add_field("dense_vector", DataType.FLOAT_VECTOR, dim=dim)
    schema.add_field("sparse_vector", DataType.SPARSE_FLOAT_VECTOR)
    for f, length in [("content_type", 32), ("document_title", 512), ("product_name", 256), ("product_code", 64),
                      ("institution_name", 256), ("risk_level", 32), ("industry", 64), ("market", 64),
                      ("publish_date", 32), ("entry_name", 1024), ("source_file", 512), ("source_path", 1024)]:
        schema.add_field(f, DataType.VARCHAR, max_length=length)
    schema.add_function(_bm25_function("search_text"))
    client.create_collection(name, schema=schema,
                             index_params=_index_params(client, ["doc_id", "content_type"]))


def _create_doc_collection(client: MilvusClient, name: str, dim: int) -> None:
    schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field("doc_id", DataType.VARCHAR, max_length=32, is_primary=True)
    for f, length in [("document_title", 512), ("product_name", 256), ("product_code", 64),
                      ("institution_name", 256), ("content_type", 32), ("publish_date", 32)]:
        schema.add_field(f, DataType.VARCHAR, max_length=length)
    schema.add_field("entity_text", DataType.VARCHAR, max_length=2048, enable_analyzer=True,
                     analyzer_params=CHINESE_ANALYZER)
    schema.add_field("dense_vector", DataType.FLOAT_VECTOR, dim=dim)
    schema.add_field("sparse_vector", DataType.SPARSE_FLOAT_VECTOR)
    schema.add_function(_bm25_function("entity_text"))
    client.create_collection(name, schema=schema, index_params=_index_params(client, ["content_type"]))


def ensure_collections() -> None:
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
    desc = client.describe_collection(name)
    for field in desc["fields"]:
        if field["name"] == "dense_vector":
            actual = field["params"].get("dim")
            if int(actual) != dim:
                raise RuntimeError(f"集合 {name} 向量维度为 {actual}，与配置 EMBEDDING_DIM={dim} 不一致")


# ---------- 写入 ----------

def delete_doc(doc_id: str) -> None:
    s, client = get_settings(), get_client()
    expr = f'doc_id == "{doc_id}"'
    client.delete(s.milvus_chunk_collection, filter=expr)
    client.delete(s.milvus_doc_collection, filter=expr)


def insert_chunks(rows: list[dict[str, Any]], batch_size: int = 200) -> None:
    s, client = get_settings(), get_client()
    for i in range(0, len(rows), batch_size):
        client.insert(s.milvus_chunk_collection, rows[i:i + batch_size])


def upsert_doc(row: dict[str, Any]) -> None:
    s = get_settings()
    get_client().upsert(s.milvus_doc_collection, [row])


# ---------- 检索 ----------

def _flatten(hits: list) -> list[dict[str, Any]]:
    results = []
    for hit in hits[0] if hits else []:
        entity = dict(hit.get("entity", {}))
        entity["_score"] = float(hit["distance"])
        results.append(entity)
    return results


def dense_search(collection: str, vector: list[float], top_k: int, expr: str = "",
                 output_fields: list[str] | None = None) -> list[dict[str, Any]]:
    hits = get_client().search(
        collection, data=[vector], anns_field="dense_vector", limit=top_k, filter=expr,
        search_params={"metric_type": "COSINE", "params": {"ef": max(64, top_k * 2)}},
        output_fields=output_fields or CHUNK_OUTPUT_FIELDS,
    )
    return _flatten(hits)


def bm25_search(collection: str, text: str, top_k: int, expr: str = "",
                output_fields: list[str] | None = None) -> list[dict[str, Any]]:
    hits = get_client().search(
        collection, data=[text], anns_field="sparse_vector", limit=top_k, filter=expr,
        search_params={"metric_type": "BM25"},
        output_fields=output_fields or CHUNK_OUTPUT_FIELDS,
    )
    return _flatten(hits)


def ping() -> bool:
    get_client().list_collections()
    return True
