# -*- coding: utf-8 -*-
"""
storage.py — 本地文件系统存储（替代 MinIO）
"""
import os
import shutil
from pathlib import Path

from app.config import settings


def _ensure_dir() -> Path:
    p = Path(settings.storage_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p


def ensure_bucket() -> None:
    """兼容旧接口：确保存储目录存在。"""
    _ensure_dir()


def upload_file(local_path: str, object_name: str) -> str:
    """将文件复制到存储目录，返回 object_name。"""
    dest_dir = _ensure_dir() / os.path.dirname(object_name)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = _ensure_dir() / object_name
    shutil.copy2(local_path, str(dest_path))
    return object_name


def download_file(object_name: str, local_path: str) -> None:
    """从存储目录读取文件到本地路径。"""
    src_path = _ensure_dir() / object_name
    shutil.copy2(str(src_path), local_path)


def delete_file(object_name: str) -> None:
    """从存储目录删除文件。"""
    p = _ensure_dir() / object_name
    if p.exists():
        p.unlink()