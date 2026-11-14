"""导入服务入口（端口 8000）：资料上传、导入进度、资料列表，并托管导入页 web/import.html。

启动命令：
    uv run uvicorn app.import_app:app --port 8000
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api import documents
from app.api.common import health_router, install_error_handlers, mount_web
from app.common.log import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    """服务生命周期：yield 之前的代码在启动时执行，之后的在关闭时执行（这里关闭时无需处理）。

    启动时：创建 MongoDB 索引、MinIO 桶、Milvus 集合（都是幂等操作，已存在则跳过），
    并把上次服务停止时没跑完的导入任务标记为失败，再清扫 temp/ 下遗留的临时目录。
    """
    from app.infra import mongo, object_store, vector_store
    from app.services.import_service import recover_interrupted, sweep_temp

    mongo.ensure_indexes()
    object_store.ensure_bucket()
    vector_store.ensure_collections()
    n = recover_interrupted()
    if n:
        logger.warning("已将 %d 个中断的导入任务标记为失败", n)
    swept = sweep_temp()
    if swept:
        logger.info("已清扫 %d 个遗留的临时目录", swept)
    yield


app = FastAPI(title="FinSage 导入服务", lifespan=lifespan)
install_error_handlers(app)
app.include_router(documents.router)
app.include_router(health_router())
mount_web(app, "import.html")  # 访问 http://127.0.0.1:8000/ 跳转到导入页
