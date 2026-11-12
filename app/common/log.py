"""日志初始化。两个服务入口（import_app / query_app）启动时各调用一次 `setup_logging()`。"""
import logging

from app.config import get_settings

# 是否已经初始化过，防止重复调用时重复添加日志处理器
_configured = False


def setup_logging() -> None:
    """配置根日志：级别取自配置项 LOG_LEVEL，输出格式为 "时间 级别 [模块名] 内容"。"""
    global _configured
    if _configured:
        return
    logging.basicConfig(
        level=get_settings().log_level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    # 第三方库日志降噪：这些库在 INFO 级别会打印大量请求明细，只保留 WARNING 及以上
    for name in ("httpx", "httpcore", "pymongo", "urllib3", "sentence_transformers"):
        logging.getLogger(name).setLevel(logging.WARNING)
    _configured = True
