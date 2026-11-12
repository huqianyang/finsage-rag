import logging

from app.config import get_settings

_configured = False


def setup_logging() -> None:
    global _configured
    if _configured:
        return
    logging.basicConfig(
        level=get_settings().log_level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    # 第三方库日志降噪
    for name in ("httpx", "httpcore", "pymongo", "urllib3", "sentence_transformers"):
        logging.getLogger(name).setLevel(logging.WARNING)
    _configured = True
