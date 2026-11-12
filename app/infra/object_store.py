"""MinIO 对象存储（见 数据库设计 §4）。桶私有，访问走临时签名 URL。"""
from datetime import timedelta
from functools import lru_cache
from pathlib import Path

from minio import Minio

from app.config import get_settings


@lru_cache
def get_client() -> Minio:
    s = get_settings()
    return Minio(s.minio_endpoint, access_key=s.minio_access_key, secret_key=s.minio_secret_key, secure=s.minio_secure)


def ensure_bucket() -> None:
    client, bucket = get_client(), get_settings().minio_bucket
    if not client.bucket_exists(bucket):
        client.make_bucket(bucket)


def raw_object_name(doc_id: str, file_name: str) -> str:
    return f"raw/{doc_id}/{file_name}"


def image_object_name(doc_id: str, image_name: str) -> str:
    return f"images/{doc_id}/{image_name}"


def upload_file(object_name: str, local_path: Path, content_type: str = "application/octet-stream") -> None:
    get_client().fput_object(get_settings().minio_bucket, object_name, str(local_path), content_type=content_type)


def remove_prefix(prefix: str) -> int:
    client, bucket = get_client(), get_settings().minio_bucket
    names = [o.object_name for o in client.list_objects(bucket, prefix=prefix, recursive=True)]
    for name in names:
        client.remove_object(bucket, name)
    return len(names)


def presigned_url(object_name: str, hours: int = 1) -> str:
    return get_client().presigned_get_object(get_settings().minio_bucket, object_name, expires=timedelta(hours=hours))


def ping() -> bool:
    get_client().bucket_exists(get_settings().minio_bucket)
    return True
