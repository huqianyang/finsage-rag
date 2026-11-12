"""业务异常与错误码（对应 API 设计 §0.1）。"""


class AppError(Exception):
    """可直接返回给前端的业务异常。"""

    def __init__(self, code: str, message: str, http_status: int = 400, detail: str | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.http_status = http_status
        self.detail = detail


class NotFoundError(AppError):
    def __init__(self, message: str = "资源不存在"):
        super().__init__("NOT_FOUND", message, 404)


class DependencyError(AppError):
    """外部依赖（Milvus / Mongo / MinIO / 模型 / MinerU）调用失败。"""

    def __init__(self, code: str, message: str, detail: str | None = None):
        super().__init__(code, message, 503, detail)


class StepError(Exception):
    """导入流程某个步骤失败，message 面向用户。"""

    def __init__(self, step: str, message: str):
        super().__init__(message)
        self.step = step
        self.message = message
