from datetime import datetime

from sqlalchemy import Column, Integer, String, Text, DateTime, Boolean, JSON, ForeignKey
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


class AuditMixin:
    """Mixin that adds standard audit columns to a model."""
    created_by = Column(String(64), nullable=True, comment="创建人")
    created_at = Column(DateTime, default=datetime.now, nullable=False, comment="创建时间")
    updated_by = Column(String(64), nullable=True, comment="修改人")
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now, nullable=True, comment="修改时间")
    deleted_by = Column(String(64), nullable=True, comment="删除人")
    deleted_at = Column(DateTime, nullable=True, comment="删除时间")
    is_deleted = Column(Boolean, default=False, nullable=False, comment="删除标记")


class User(Base, AuditMixin):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(64), unique=True, nullable=False)
    password_hash = Column(String(256), nullable=False)
    role = Column(String(16), nullable=False, default="external", comment="admin/internal/external")


class Session(Base, AuditMixin):
    __tablename__ = "sessions"
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    title = Column(String(256), nullable=True)


class Message(Base, AuditMixin):
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(Integer, ForeignKey("sessions.id"), nullable=False)
    role = Column(String(16), nullable=False, comment="user/assistant")
    content = Column(Text, nullable=False)
    intents = Column(JSON, nullable=True)


class IntentLog(Base, AuditMixin):
    __tablename__ = "intent_logs"
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(Integer, nullable=True)
    question = Column(Text, nullable=False)
    failed_level = Column(String(4), nullable=False, comment="L1/L2/L3")
    scores = Column(JSON, nullable=True)
    final_intent = Column(String(64), nullable=True)


class Document(Base, AuditMixin):
    __tablename__ = "documents"
    id = Column(Integer, primary_key=True, autoincrement=True)
    filename = Column(String(256), nullable=False)
    category = Column(String(32), nullable=False, comment="legal/tender/product")
    file_path = Column(String(512), nullable=False, comment="MinIO key")
    file_size = Column(Integer, nullable=True)
    status = Column(String(16), default="pending", comment="pending/processing/done/failed")
    uploaded_by = Column(Integer, ForeignKey("users.id"), nullable=True)


class IndexTask(Base, AuditMixin):
    __tablename__ = "index_tasks"
    id = Column(Integer, primary_key=True, autoincrement=True)
    task_type = Column(String(16), nullable=False, comment="full/incremental")
    category = Column(String(32), nullable=False)
    document_ids = Column(JSON, nullable=True)
    status = Column(String(16), default="queued", comment="queued/running/done/failed")
    result_msg = Column(Text, nullable=True)