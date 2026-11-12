"""本地模型：bge-m3（文本向量）与 bge-reranker-large（重排），运行在本机 GPU。

- 两个模型都是进程内单例，首次调用时才加载（加载约 5~30 秒），之后常驻显存。
  查询服务启动时会在后台线程预热，避免第一个问题等待加载。
- torch / sentence_transformers 体积大、导入慢，所以放在函数内部延迟导入，
  不需要本地模型的代码路径（如单元测试）不会被拖慢。
"""
import logging
import threading
from functools import lru_cache

from app.config import get_settings

logger = logging.getLogger(__name__)
# GPU 推理串行化：多个请求线程同时推理会争抢显存，可能导致显存溢出，这里加锁一次只跑一个批次
_gpu_lock = threading.Lock()


def _device() -> str:
    """确定推理设备：配置为 cuda 但当前环境没有可用 GPU 时，自动退回 CPU 并打印警告。"""
    import torch

    want = get_settings().model_device
    if want == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA 不可用，本地模型改用 CPU 运行")
        return "cpu"
    return want


@lru_cache
def get_embedder():
    """加载 bge-m3 向量模型（只加载一次）。"""
    from sentence_transformers import SentenceTransformer

    s = get_settings()
    device = _device()
    logger.info("加载向量模型 %s（%s）", s.embedding_model_path, device)
    model = SentenceTransformer(s.embedding_model_path, device=device)
    if device == "cuda":
        # 半精度（fp16）：显存占用减半、速度更快，对检索效果影响可以忽略；CPU 不支持 fp16 推理
        model.half()
    return model


@lru_cache
def get_reranker():
    """加载 bge-reranker-large 重排模型（只加载一次）。

    重排模型是 Cross-Encoder：把"问题 + 片段"拼在一起输入，直接输出相关度分数，
    比向量相似度更准，但每对都要单独算一次，所以只用于对召回的少量候选（约 20 条）排序。
    """
    from sentence_transformers import CrossEncoder

    s = get_settings()
    device = _device()
    logger.info("加载重排模型 %s（%s）", s.reranker_model_path, device)
    # max_length：问题 + 片段的总 token 数上限，超出部分会被截断
    model = CrossEncoder(s.reranker_model_path, max_length=s.reranker_max_length, device=device)
    if device == "cuda":
        model.model.half()
    return model


def embed(texts: list[str], batch_size: int = 16) -> list[list[float]]:
    """把一批文本编码成稠密向量。

    返回 L2 归一化后的向量（长度为 1），此时余弦相似度等于点积，
    与 Milvus 集合的 COSINE 度量配合使用。
    """
    if not texts:
        return []
    model = get_embedder()
    with _gpu_lock:
        vectors = model.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                               convert_to_numpy=True, show_progress_bar=False)
    # Milvus 的 FLOAT_VECTOR 需要 float32；fp16 推理出来的结果先转回 float32 再转成普通列表
    return vectors.astype("float32").tolist()


def rerank(query: str, passages: list[str], batch_size: int = 16) -> list[float]:
    """计算问题与每个片段的相关度分数，返回顺序与 passages 一致。

    分数经过 sigmoid，范围 0~1：明显相关的片段通常在 0.5 以上，无关的接近 0。
    """
    if not passages:
        return []
    model = get_reranker()
    with _gpu_lock:
        scores = model.predict([(query, p) for p in passages], batch_size=batch_size,
                               show_progress_bar=False, convert_to_numpy=True)
    return [float(x) for x in scores]
