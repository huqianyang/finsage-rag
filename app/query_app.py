"""查询服务入口（端口 8001）：流式问答、会话管理、查看原文，并托管问答页 web/chat.html。

启动命令：
    uv run uvicorn app.query_app:app --port 8001
"""
import logging
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api import chat
from app.api.common import health_router, install_error_handlers, mount_web
from app.common.log import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


def _warm_up() -> None:
    """后台预加载本地模型，避免第一个问题等待模型加载。

    向量模型和重排模型首次加载需要数秒到数十秒；用一条假数据各调用一次，
    触发加载并完成 GPU 初始化。失败只记日志，不影响服务启动（问答时还会再尝试加载）。
    """
    try:
        from app.infra import local_models

        local_models.embed(["预热"])
        local_models.rerank("预热", ["预热"])
        logger.info("本地模型预热完成")
    except Exception:
        logger.exception("本地模型预热失败")


@asynccontextmanager
async def lifespan(_: FastAPI):
    """启动时：确保 MongoDB 索引和 Milvus 集合存在，并在后台线程预热本地模型（不阻塞服务启动）。"""
    from app.infra import mongo, vector_store

    mongo.ensure_indexes()
    vector_store.ensure_collections()
    threading.Thread(target=_warm_up, daemon=True).start()
    yield


app = FastAPI(title="FinSage 查询服务", lifespan=lifespan)
install_error_handlers(app)
app.include_router(chat.router)
app.include_router(health_router())
mount_web(app, "chat.html")  # 访问 http://127.0.0.1:8001/ 跳转到问答页
