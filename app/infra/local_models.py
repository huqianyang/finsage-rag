"""本地模型：bge-m3（向量）与 bge-reranker-large（重排），进程内单例、首次调用时加载。"""
import logging
import threading
from functools import lru_cache

from app.config import get_settings

logger = logging.getLogger(__name__)
# GPU 推理串行化，避免多线程同时占用显存
_gpu_lock = threading.Lock()


def _device() -> str:
    import torch

    want = get_settings().model_device
    if want == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA 不可用，本地模型改用 CPU 运行")
        return "cpu"
    return want


@lru_cache
def get_embedder():
    from sentence_transformers import SentenceTransformer

    s = get_settings()
    device = _device()
    logger.info("加载向量模型 %s（%s）", s.embedding_model_path, device)
    model = SentenceTransformer(s.embedding_model_path, device=device)
    if device == "cuda":
        model.half()
    return model


@lru_cache
def get_reranker():
    from sentence_transformers import CrossEncoder

    s = get_settings()
    device = _device()
    logger.info("加载重排模型 %s（%s）", s.reranker_model_path, device)
    model = CrossEncoder(s.reranker_model_path, max_length=s.reranker_max_length, device=device)
    if device == "cuda":
        model.model.half()
    return model


def embed(texts: list[str], batch_size: int = 16) -> list[list[float]]:
    """返回 L2 归一化后的稠密向量。"""
    if not texts:
        return []
    model = get_embedder()
    with _gpu_lock:
        vectors = model.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                               convert_to_numpy=True, show_progress_bar=False)
    return vectors.astype("float32").tolist()


def rerank(query: str, passages: list[str], batch_size: int = 16) -> list[float]:
    """返回每个 passage 的相关度分数（0~1，sigmoid 后）。"""
    if not passages:
        return []
    model = get_reranker()
    with _gpu_lock:
        scores = model.predict([(query, p) for p in passages], batch_size=batch_size,
                               show_progress_bar=False, convert_to_numpy=True)
    return [float(x) for x in scores]
