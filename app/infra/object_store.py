"""MinIO 对象存储（数据库设计文档 §4）。

存储内容：
- raw/{doc_id}/{文件名}       用户上传的原始文件，用于"查看原文"
- images/{doc_id}/{图片名}    文档解析出的图片

桶是私有的，浏览器不能直接访问；查看原文时由查询服务生成有时效的临时签名 URL。
"""
from datetime import timedelta
from functools import lru_cache
from pathlib import Path

from minio import Minio

from app.config import get_settings


@lru_cache
def get_client() -> Minio:
    """创建（并缓存）MinIO 客户端。"""
    s = get_settings()
    return Minio(s.minio_endpoint, access_key=s.minio_access_key, secret_key=s.minio_secret_key, secure=s.minio_secure)


def ensure_bucket() -> None:
    """业务桶不存在时创建。导入服务启动时调用。"""
    client, bucket = get_client(), get_settings().minio_bucket
    if not client.bucket_exists(bucket):
        client.make_bucket(bucket)


def raw_object_name(doc_id: str, file_name: str) -> str:
    """原始文件在桶内的对象路径。"""
    return f"raw/{doc_id}/{file_name}"


def image_object_name(doc_id: str, image_name: str) -> str:
    """文档图片在桶内的对象路径。"""
    return f"images/{doc_id}/{image_name}"


def upload_file(object_name: str, local_path: Path, content_type: str = "application/octet-stream") -> None:
    """把本地文件上传到桶内指定路径（同名对象会被覆盖）。"""
    get_client().fput_object(get_settings().minio_bucket, object_name, str(local_path), content_type=content_type)


def remove_prefix(prefix: str) -> int:
    """删除某个前缀（"目录"）下的全部对象，返回删除数量。覆盖导入时用来清理旧文件和旧图片。"""
    client, bucket = get_client(), get_settings().minio_bucket
    # 先把对象名全部列出来再删，避免边遍历边删除
    names = [o.object_name for o in client.list_objects(bucket, prefix=prefix, recursive=True)]
    for name in names:
        client.remove_object(bucket, name)
    return len(names)


def presigned_url(object_name: str, hours: int = 1) -> str:
    """生成临时下载链接，默认 1 小时内有效，过期后需重新生成。"""
    return get_client().presigned_get_object(get_settings().minio_bucket, object_name, expires=timedelta(hours=hours))


def ping() -> bool:
    """健康检查：连不上或鉴权失败会抛异常。"""
    get_client().bucket_exists(get_settings().minio_bucket)
    return True
