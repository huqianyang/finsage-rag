"""导入服务和查询服务共用的 Web 层组件：全局异常处理、健康检查、静态页面挂载。"""
import logging

from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from app.common.errors import AppError
from app.config import PROJECT_ROOT

logger = logging.getLogger(__name__)


def install_error_handlers(app: FastAPI) -> None:
    """注册全局异常处理器，统一错误响应格式为 {"code", "message", "detail"}（API 设计 §0）。"""

    @app.exception_handler(AppError)
    async def _app_error(_: Request, e: AppError):
        """业务异常：按异常自带的状态码和错误码返回。"""
        return JSONResponse(status_code=e.http_status,
                            content={"code": e.code, "message": e.message, "detail": e.detail})

    @app.exception_handler(Exception)
    async def _unknown(_: Request, e: Exception):
        """未预料的异常：记录完整堆栈，返回 500 和通用提示（具体原因放在 detail 里便于排查）。"""
        logger.exception("未处理异常")
        return JSONResponse(status_code=500,
                            content={"code": "INTERNAL_ERROR", "message": "服务内部错误，请稍后重试", "detail": str(e)})


def health_router() -> APIRouter:
    """健康检查路由：GET /api/health。"""
    router = APIRouter()

    @router.get("/api/health")
    def health():
        """依次检查 Milvus、MongoDB、MinIO 是否可用；全部正常返回 200，否则返回 503 和各依赖状态。"""
        from app.infra import mongo, object_store, vector_store

        deps = {}
        for name, fn in [("milvus", vector_store.ping), ("mongo", mongo.ping), ("minio", object_store.ping)]:
            try:
                fn()
                deps[name] = "ok"
            except Exception as e:
                # 只返回异常类型，不返回详细信息（可能包含地址等内部信息）
                deps[name] = f"error: {type(e).__name__}"
        ok = all(v == "ok" for v in deps.values())
        return JSONResponse(status_code=200 if ok else 503,
                            content={"status": "ok" if ok else "degraded", "deps": deps})

    return router


def mount_web(app: FastAPI, index_page: str) -> None:
    """托管前端静态页面：web/ 目录挂到 /web 路径下，访问根路径 / 时跳转到本服务的主页面。"""
    app.mount("/web", StaticFiles(directory=PROJECT_ROOT / "web"), name="web")

    @app.get("/", include_in_schema=False)  # include_in_schema=False：不在 /docs 接口文档中显示
    def index():
        return RedirectResponse(f"/web/{index_page}")
