"""全局配置：从 .env 与环境变量读取，进程内单例。"""
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=PROJECT_ROOT / ".env", env_file_encoding="utf-8", extra="ignore")

    # 大模型（硅基流动）
    siliconflow_api_key: str = ""
    siliconflow_base_url: str = "https://api.siliconflow.cn/v1"
    llm_model: str = "deepseek-ai/DeepSeek-V4.1-Flash"
    vlm_model: str = "deepseek-ai/DeepSeek-V4.1-Flash"
    llm_timeout_seconds: int = 60
    llm_enable_thinking: bool = False

    # 本地模型
    embedding_model_path: str = "D:/ai_models/modelscope_cache/models/BAAI/bge-m3"
    embedding_dim: int = 1024
    reranker_model_path: str = "D:/ai_models/rerank/rerank/BAAI/bge-reranker-large"
    reranker_max_length: int = 512
    model_device: str = "cuda"

    # MinerU
    mineru_api_token: str = ""
    mineru_base_url: str = "https://mineru.net/api/v4"
    mineru_model_version: str = "vlm"
    mineru_poll_interval_seconds: int = 5
    mineru_timeout_seconds: int = 900

    # 联网搜索（百炼 MCP）
    dashscope_api_key: str = ""
    web_search_mcp_url: str = "https://dashscope.aliyuncs.com/api/v1/mcps/WebSearch/mcp"
    web_search_top_k: int = 5

    # Milvus
    milvus_uri: str = "http://192.168.10.100:19530"
    milvus_chunk_collection: str = "fin_chunks"
    milvus_doc_collection: str = "fin_docs"

    # MongoDB
    mongo_host: str = "192.168.10.100"
    mongo_port: int = 27017
    mongo_user: str = ""
    mongo_password: str = ""
    mongo_auth_source: str = "admin"
    mongo_db: str = "finsage"

    # MinIO
    minio_endpoint: str = "192.168.10.100:9000"
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = ""
    minio_bucket: str = "finsage"
    minio_secure: bool = False

    # 导入
    upload_max_mb: int = 50
    import_workers: int = 2
    chunk_target_chars: int = 500
    chunk_min_chars: int = 300
    chunk_max_chars: int = 800

    # 检索与问答
    history_turns: int = 5
    dense_top_k: int = 20
    bm25_top_k: int = 20
    rrf_k: int = 60
    rrf_top_n: int = 20
    rerank_top_n: int = 6
    web_max_in_context: int = 2
    no_answer_score_threshold: float = 0.3
    entity_high_confidence: float = 0.75
    entity_mid_confidence: float = 0.45
    entity_score_gap: float = 0.08
    entity_max_options: int = 3

    log_level: str = "INFO"

    @property
    def temp_dir(self) -> Path:
        path = PROJECT_ROOT / "temp"
        path.mkdir(exist_ok=True)
        return path


@lru_cache
def get_settings() -> Settings:
    return Settings()
