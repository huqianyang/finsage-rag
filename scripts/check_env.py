"""环境连通性检查：中间件、本地模型、云端模型、MinerU、联网搜索。

用法：uv run python -m scripts.check_env [--skip-mineru]
只打印是否可用，不打印任何密钥。
"""
import sys
import time

from app.config import get_settings


def step(name, fn):
    t = time.time()
    try:
        msg = fn()
        print(f"[ OK ] {name}：{msg or ''} ({time.time() - t:.1f}s)")
        return True
    except Exception as e:
        print(f"[FAIL] {name}：{type(e).__name__}: {e}")
        return False


def check_mongo():
    from app.infra import mongo
    mongo.ping()
    mongo.ensure_indexes()
    return f"库 {get_settings().mongo_db}，索引已创建"


def check_minio():
    from app.infra import object_store
    object_store.ensure_bucket()
    return f"桶 {get_settings().minio_bucket} 就绪"


def check_milvus_analyzer():
    """在临时集合上验证 BM25 + 中文分词器可用。"""
    from pymilvus import DataType, MilvusClient
    from app.infra.vector_store import CHINESE_ANALYZER, _bm25_function, get_client

    c = get_client()
    name = "finsage_probe"
    if c.has_collection(name):
        c.drop_collection(name)
    schema = MilvusClient.create_schema(auto_id=True)
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("search_text", DataType.VARCHAR, max_length=1024, enable_analyzer=True,
                     analyzer_params=CHINESE_ANALYZER)
    schema.add_field("sparse_vector", DataType.SPARSE_FLOAT_VECTOR)
    schema.add_function(_bm25_function("search_text"))
    idx = c.prepare_index_params()
    idx.add_index(field_name="sparse_vector", index_type="SPARSE_INVERTED_INDEX", metric_type="BM25")
    c.create_collection(name, schema=schema, index_params=idx)
    try:
        c.insert(name, [{"search_text": "华夏债券投资基金的风险等级为中低风险"},
                        {"search_text": "招商银行第一季度营业收入同比变化"}])
        c.flush(name)
        c.load_collection(name)
        hits = c.search(name, data=["招商银行营收"], anns_field="sparse_vector", limit=2,
                        search_params={"metric_type": "BM25"}, output_fields=["search_text"])
        top = hits[0][0]["entity"]["search_text"]
        assert "招商银行" in top, f"BM25 排序不符合预期：{top}"
        return f"中文 BM25 命中：{top}"
    finally:
        c.drop_collection(name)


def check_milvus_collections():
    from app.infra import vector_store
    vector_store.ensure_collections()
    s = get_settings()
    return f"{s.milvus_chunk_collection}、{s.milvus_doc_collection} 就绪"


def check_embedding():
    from app.infra.local_models import embed
    v = embed(["什么是净值型理财"])
    assert len(v[0]) == get_settings().embedding_dim, f"维度 {len(v[0])}"
    return f"维度 {len(v[0])}"


def check_reranker():
    from app.infra.local_models import rerank
    s = rerank("基金赎回多久到账", ["基金赎回款一般在T+1至T+7个交易日内到账", "贵州茅台一季度营收增长"])
    assert s[0] > s[1], s
    return f"分数 {[round(x, 3) for x in s]}"


def check_llm():
    from app.infra.llm import chat
    return chat([{"role": "user", "content": "只回复两个字：你好"}], max_tokens=10).strip()


def check_web_search():
    from app.infra import web_search
    if not get_settings().dashscope_api_key:
        return "未配置 DASHSCOPE_API_KEY，已跳过"
    pages = web_search.search("中国人民银行 货币政策执行报告")
    if not pages:
        raise RuntimeError("无结果，查看上方日志")
    return f"{len(pages)} 条，首条：{pages[0]['title']}"


def check_mineru():
    """只检查鉴权（申请上传地址），不真正上传文件、不消耗解析额度。"""
    if not get_settings().mineru_api_token:
        return "未配置 MINERU_API_TOKEN，已跳过"
    import httpx
    s = get_settings()
    r = httpx.post(f"{s.mineru_base_url}/file-urls/batch",
                   headers={"Authorization": f"Bearer {s.mineru_api_token}"},
                   json={"files": [{"name": "probe.pdf"}], "model_version": s.mineru_model_version}, timeout=30)
    body = r.json()
    if body.get("code") != 0:
        raise RuntimeError(f"code={body.get('code')} msg={body.get('msg')}")
    return "鉴权通过，可申请上传地址"


if __name__ == "__main__":
    checks = [("MongoDB", check_mongo), ("MinIO", check_minio), ("Milvus 中文 BM25", check_milvus_analyzer),
              ("Milvus 集合", check_milvus_collections), ("本地向量模型", check_embedding),
              ("本地重排模型", check_reranker), ("硅基流动 LLM", check_llm), ("百炼联网搜索", check_web_search),
              ("MinerU", check_mineru)]
    ok = all([step(n, f) for n, f in checks])
    sys.exit(0 if ok else 1)
