"""MongoDB 客户端与集合（见 数据库设计 §3）。"""
from functools import lru_cache
from urllib.parse import quote_plus

from pymongo import ASCENDING, DESCENDING, MongoClient
from pymongo.database import Database

from app.config import get_settings


@lru_cache
def get_client() -> MongoClient:
    s = get_settings()
    if s.mongo_user:
        uri = (
            f"mongodb://{quote_plus(s.mongo_user)}:{quote_plus(s.mongo_password)}"
            f"@{s.mongo_host}:{s.mongo_port}/?authSource={s.mongo_auth_source}"
        )
    else:
        uri = f"mongodb://{s.mongo_host}:{s.mongo_port}/"
    return MongoClient(uri, serverSelectionTimeoutMS=5000, tz_aware=True)


def get_db() -> Database:
    return get_client()[get_settings().mongo_db]


def documents():
    return get_db()["documents"]


def import_tasks():
    return get_db()["import_tasks"]


def sessions():
    return get_db()["sessions"]


def messages():
    return get_db()["messages"]


def ensure_indexes() -> None:
    documents().create_index("file_name", unique=True)
    documents().create_index([("content_type", ASCENDING), ("updated_at", DESCENDING)])
    import_tasks().create_index([("doc_id", ASCENDING), ("created_at", DESCENDING)])
    import_tasks().create_index([("status", ASCENDING), ("created_at", DESCENDING)])
    sessions().create_index([("client_id", ASCENDING), ("updated_at", DESCENDING)])
    messages().create_index([("session_id", ASCENDING), ("created_at", ASCENDING)])


def ping() -> bool:
    get_client().admin.command("ping")
    return True
