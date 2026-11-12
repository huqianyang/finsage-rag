"""全局配置。

所有可调参数集中在这里，统一从项目根目录的 `.env` 文件和进程环境变量读取
（环境变量优先级高于 `.env`）。字段名与环境变量名一一对应，大小写不敏感，
例如字段 `llm_model` 对应环境变量 `LLM_MODEL`。

使用方式：任何模块都通过 `get_settings()` 获取同一个配置对象（进程内单例），
不要直接实例化 `Settings`，避免重复读取文件。
"""
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# 项目根目录：本文件位于 app/config.py，向上两级即为 finsage-rag/
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """应用配置项。默认值仅用于本机开发，密钥类字段默认为空，必须在 `.env` 中填写。"""

    # env_file：指定 .env 路径；extra="ignore"：.env 中存在未定义的变量时忽略而不是报错
    model_config = SettingsConfigDict(env_file=PROJECT_ROOT / ".env", env_file_encoding="utf-8", extra="ignore")

    # ==================== 大模型（硅基流动，OpenAI 兼容接口） ====================
    siliconflow_api_key: str = ""  # 硅基流动 API Key
    siliconflow_base_url: str = "https://api.siliconflow.cn/v1"  # OpenAI 兼容接口地址
    llm_model: str = "deepseek-ai/DeepSeek-V4.1-Flash"  # 问答生成、问题改写、元数据抽取使用的模型
    vlm_model: str = "deepseek-ai/DeepSeek-V4.1-Flash"  # 图片摘要使用的视觉模型
    llm_timeout_seconds: int = 60  # 单次大模型请求超时（秒）
    llm_enable_thinking: bool = False  # 是否开启思考模式；开启会明显增加首字延迟（实测 0.5s → 2.4s+）

    # ==================== 本地模型（向量与重排，运行在本机 GPU） ====================
    embedding_model_path: str = "D:/ai_models/modelscope_cache/models/BAAI/bge-m3"  # bge-m3 模型目录
    embedding_dim: int = 1024  # bge-m3 输出向量维度，必须与 Milvus 集合的 dense_vector 维度一致
    reranker_model_path: str = "D:/ai_models/rerank/rerank/BAAI/bge-reranker-large"  # 重排模型目录
    reranker_max_length: int = 512  # 重排模型单次输入上限（问题 + 片段的 token 数），超出截断
    model_device: str = "cuda"  # 推理设备：cuda / cpu；cuda 不可用时自动退回 cpu

    # ==================== 文档解析（MinerU 云 API） ====================
    mineru_api_token: str = ""  # MinerU 控制台申请的 API Token
    mineru_base_url: str = "https://mineru.net/api/v4"  # 精准解析 API 地址
    mineru_model_version: str = "vlm"  # 解析模型：pipeline / vlm
    mineru_poll_interval_seconds: int = 5  # 轮询解析结果的间隔（秒）
    mineru_timeout_seconds: int = 900  # 单个文件解析的最长等待时间（秒）

    # ==================== 联网搜索（阿里云百炼 WebSearch MCP） ====================
    dashscope_api_key: str = ""  # 百炼 API Key；为空时自动跳过联网搜索
    web_search_mcp_url: str = "https://dashscope.aliyuncs.com/api/v1/mcps/WebSearch/mcp"  # MCP 服务地址
    web_search_top_k: int = 5  # 每次联网搜索返回的结果条数

    # ==================== Milvus 向量库 ====================
    milvus_uri: str = "http://192.168.10.100:19530"
    milvus_chunk_collection: str = "fin_chunks"  # 切片集合：存每个文档片段及其向量
    milvus_doc_collection: str = "fin_docs"  # 文档集合：每份资料一条，用于实体确认

    # ==================== MongoDB（资料、导入任务、会话、消息） ====================
    mongo_host: str = "192.168.10.100"
    mongo_port: int = 27017
    mongo_user: str = ""  # 为空表示 MongoDB 未开启鉴权
    mongo_password: str = ""
    mongo_auth_source: str = "admin"  # 账号所在的认证库
    mongo_db: str = "finsage"  # 业务数据库名

    # ==================== MinIO 对象存储（原文件、图片） ====================
    minio_endpoint: str = "192.168.10.100:9000"  # 主机:端口，不带 http://
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = ""
    minio_bucket: str = "finsage"  # 业务桶，启动时不存在会自动创建
    minio_secure: bool = False  # 是否使用 HTTPS

    # ==================== 资料导入 ====================
    upload_max_mb: int = 50  # 单个上传文件大小上限（MB）
    import_workers: int = 2  # 同时执行的导入任务数（后台线程数）
    chunk_target_chars: int = 500  # 切片目标长度（字）
    chunk_min_chars: int = 300  # 短于此长度的章节会与下一节合并
    chunk_max_chars: int = 800  # 长于此长度的章节会二次切分

    # ==================== 检索与问答 ====================
    history_turns: int = 5  # 问题改写与生成时参考的最近对话轮数
    dense_top_k: int = 20  # 稠密向量检索召回条数
    bm25_top_k: int = 20  # BM25 关键词检索召回条数
    rrf_k: int = 60  # RRF 融合公式 1/(k+rank) 中的平滑常数 k
    rrf_top_n: int = 20  # RRF 融合后保留的条数（进入重排）
    rerank_top_n: int = 6  # 重排后进入大模型上下文的条数
    web_max_in_context: int = 2  # 上下文中网络来源最多条数，避免网页挤掉知识库资料
    no_answer_score_threshold: float = 0.3  # 重排分数低于该值视为不相关；知识库全部低于它即判定"无资料"
    entity_high_confidence: float = 0.75  # 实体匹配：最高分 ≥ 该值且与第二名差距足够 → 直接锁定
    entity_mid_confidence: float = 0.45  # 实体匹配：≥ 该值的候选有多个 → 请用户澄清
    entity_score_gap: float = 0.08  # 实体匹配：第一名与第二名的最小分差
    entity_max_options: int = 3  # 澄清时最多列出的候选数

    log_level: str = "INFO"  # 日志级别：DEBUG / INFO / WARNING / ERROR

    @property
    def temp_dir(self) -> Path:
        """导入时存放上传文件和解析结果的临时目录（项目根目录下的 temp/），不存在时自动创建。"""
        path = PROJECT_ROOT / "temp"
        path.mkdir(exist_ok=True)
        return path


@lru_cache
def get_settings() -> Settings:
    """返回全局唯一的配置对象。lru_cache 保证只在第一次调用时读取 .env。"""
    return Settings()
