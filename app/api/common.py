"""两个服务共用：异常处理、健康检查、静态页面挂载。"""
import logging

from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from app.common.errors import AppError
from app.config import PROJECT_ROOT

logger = logging.getLogger(__name__)


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def _app_error(_: Request, e: AppError):
        return JSONResponse(status_code=e.http_status,
                            content={"code": e.code, "message": e.message, "detail": e.detail})

    @app.exception_handler(Exception)
    async def _unknown(_: Request, e: Exception):
        logger.exception("未处理异常")
        return JSONResponse(status_code=500,
                            content={"code": "INTERNAL_ERROR", "message": "服务内部错误，请稍后重试", "detail": str(e)})


def health_router() -> APIRouter:
    router = APIRouter()

    @router.get("/api/health")
    def health():
        from app.infra import mongo, object_store, vector_store

        deps = {}
        for name, fn in [("milvus", vector_store.ping), ("mongo", mongo.ping), ("minio", object_store.ping)]:
            try:
                fn()
                deps[name] = "ok"
            except Exception as e:
                deps[name] = f"error: {type(e).__name__}"
        ok = all(v == "ok" for v in deps.values())
        return JSONResponse(status_code=200 if ok else 503,
                            content={"status": "ok" if ok else "degraded", "deps": deps})

    return router


def mount_web(app: FastAPI, index_page: str) -> None:
    app.mount("/web", StaticFiles(directory=PROJECT_ROOT / "web"), name="web")

    @app.get("/", include_in_schema=False)
    def index():
        return RedirectResponse(f"/web/{index_page}")
