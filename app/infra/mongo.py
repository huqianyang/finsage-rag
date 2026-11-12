"""MongoDB 客户端与集合（数据库设计文档 §3）。

业务数据库（默认 finsage）中有 4 个集合：
- documents     资料登记：每份上传的文件一条，记录类型、元数据、导入状态、切片数
- import_tasks  导入任务：每次上传一条，记录 6 个步骤的进度和失败原因
- sessions      会话：按浏览器匿名 client_id 区分，记录标题和多轮追问锁定的实体
- messages      消息：用户问题和系统回答，回答中包含引用来源与检索轨迹
"""
from functools import lru_cache
from urllib.parse import quote_plus

from pymongo import ASCENDING, DESCENDING, MongoClient
from pymongo.database import Database

from app.config import get_settings


@lru_cache
def get_client() -> MongoClient:
    """创建（并缓存）MongoDB 客户端。MongoClient 自带连接池且线程安全，全进程共用一个即可。"""
    s = get_settings()
    if s.mongo_user:
        # 开启了鉴权：用户名和密码需要 URL 编码，防止其中的特殊字符（@ : / 等）破坏连接串
        uri = (
            f"mongodb://{quote_plus(s.mongo_user)}:{quote_plus(s.mongo_password)}"
            f"@{s.mongo_host}:{s.mongo_port}/?authSource={s.mongo_auth_source}"
        )
    else:
        uri = f"mongodb://{s.mongo_host}:{s.mongo_port}/"
    # serverSelectionTimeoutMS：连不上时 5 秒内报错，而不是默认的 30 秒
    # tz_aware=True：读出的 datetime 带时区（UTC），前端可正确换算为本地时间
    return MongoClient(uri, serverSelectionTimeoutMS=5000, tz_aware=True)


def get_db() -> Database:
    """返回业务数据库对象。"""
    return get_client()[get_settings().mongo_db]


def documents():
    """资料登记集合。"""
    return get_db()["documents"]


def import_tasks():
    """导入任务集合。"""
    return get_db()["import_tasks"]


def sessions():
    """会话集合。"""
    return get_db()["sessions"]


def messages():
    """消息集合。"""
    return get_db()["messages"]


def ensure_indexes() -> None:
    """创建所需索引（幂等，已存在则跳过）。服务启动时调用。"""
    # 文件名唯一：同名文件再次上传视为覆盖导入
    documents().create_index("file_name", unique=True)
    # 资料列表按类型筛选、按更新时间倒序
    documents().create_index([("content_type", ASCENDING), ("updated_at", DESCENDING)])
    # 查某份资料的导入历史
    import_tasks().create_index([("doc_id", ASCENDING), ("created_at", DESCENDING)])
    # 查进行中的任务（导入页刷新后恢复进度）
    import_tasks().create_index([("status", ASCENDING), ("created_at", DESCENDING)])
    # 会话列表：某个浏览器的会话按最近更新倒序
    sessions().create_index([("client_id", ASCENDING), ("updated_at", DESCENDING)])
    # 会话消息：按时间正序展示
    messages().create_index([("session_id", ASCENDING), ("created_at", ASCENDING)])


def ping() -> bool:
    """健康检查：连不上会抛异常。"""
    get_client().admin.command("ping")
    return True
