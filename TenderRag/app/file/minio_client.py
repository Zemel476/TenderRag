# -*- coding: utf-8 -*-
"""
minio_client.py
MinIO 客户端封装
"""
from minio import Minio
from minio.error import S3Error

from app.config import settings

_client = None


def get_minio() -> Minio:
    """返回模块级 Minio 客户端单例。"""
    global _client
    if _client is None:
        _client = Minio(
            settings.minio_endpoint,
            access_key=settings.minio_access_key,
            secret_key=settings.minio_secret_key,
            secure=settings.minio_secure,
        )
    return _client


def ensure_bucket() -> None:
    """如果 bucket 不存在则创建。"""
    client = get_minio()
    bucket = settings.minio_bucket
    if not client.bucket_exists(bucket):
        client.make_bucket(bucket)


def upload_file(local_path: str, object_name: str) -> str:
    """上传文件到 MinIO，返回 object_name。"""
    client = get_minio()
    ensure_bucket()
    client.fput_object(settings.minio_bucket, object_name, local_path)
    return object_name


def download_file(object_name: str, local_path: str) -> None:
    """从 MinIO 下载文件到本地。"""
    client = get_minio()
    client.fget_object(settings.minio_bucket, object_name, local_path)


def delete_file(object_name: str) -> None:
    """从 MinIO 删除文件。"""
    client = get_minio()
    client.remove_object(settings.minio_bucket, object_name)