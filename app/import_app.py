"""导入服务入口（:8000）。

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
    from app.infra import mongo, object_store, vector_store
    from app.services.import_service import recover_interrupted

    mongo.ensure_indexes()
    object_store.ensure_bucket()
    vector_store.ensure_collections()
    n = recover_interrupted()
    if n:
        logger.warning("已将 %d 个中断的导入任务标记为失败", n)
    yield


app = FastAPI(title="FinSage 导入服务", lifespan=lifespan)
install_error_handlers(app)
app.include_router(documents.router)
app.include_router(health_router())
mount_web(app, "import.html")
