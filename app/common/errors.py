"""业务异常与错误码（对应 API 设计文档 §0.1）。

约定：
- 需要直接告诉前端的错误抛 `AppError`（或其子类），由 `app/api/common.py` 中的全局异常处理器
  统一转换成 `{"code", "message", "detail"}` 格式的 JSON 响应。
- 导入流程中某一步失败抛 `StepError`，由导入服务捕获后写入任务的 error 字段。
"""


class AppError(Exception):
    """可直接返回给前端的业务异常。

    Args:
        code: 机器可读的错误码，如 INVALID_ARGUMENT、NOT_FOUND
        message: 给用户看的中文提示
        http_status: 返回的 HTTP 状态码，默认 400
        detail: 排查问题用的补充信息（可选）
    """

    def __init__(self, code: str, message: str, http_status: int = 400, detail: str | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.http_status = http_status
        self.detail = detail


class NotFoundError(AppError):
    """资源不存在，或不属于当前用户（例如访问别人的会话），统一返回 404。"""

    def __init__(self, message: str = "资源不存在"):
        super().__init__("NOT_FOUND", message, 404)


class DependencyError(AppError):
    """外部依赖（Milvus / Mongo / MinIO / 模型 / MinerU）调用失败，返回 503。"""

    def __init__(self, code: str, message: str, detail: str | None = None):
        super().__init__(code, message, 503, detail)


class StepError(Exception):
    """导入流程某个步骤失败。

    Args:
        step: 失败的步骤编码（parse / image / split / metadata / embed / index）
        message: 面向用户的失败原因，会显示在导入页的任务卡片上
    """

    def __init__(self, step: str, message: str):
        super().__init__(message)
        self.step = step
        self.message = message
