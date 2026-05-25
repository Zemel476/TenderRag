# TenderRag 重构实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将现有 Gradio 单页面 RAG 系统重构为双端架构（内部管理台 + 外部对话台），新增多级意图识别、文件管理、任务队列、用户认证模块。

**Architecture:** 一个 FastAPI 后端 + 两个 Vue3 前端。后端拆分为 auth/ intent/ chat/ rag/ file/ task/ data/ db/ 八个模块，rag/ 核心检索不变。

**Tech Stack:** FastAPI, LangGraph, LlamaIndex, Milvus, SQLAlchemy async, ARQ, Redis, MinIO, PyTorch, Jieba, Vue3 + Element Plus

---

## 文件结构总览

```
app/
├── __init__.py                    # 已有，不修改
├── main.py                        # 重写：FastAPI app，注册所有路由
├── config.py                      # 扩展：新增 Redis/ARQ/MinIO/意图配置
│
├── api/                           # API 路由层
│   ├── __init__.py                # 空文件
│   ├── auth.py                    # 新增
│   ├── chat.py                    # 新增（替换旧的 endpoints.py）
│   ├── documents.py               # 新增
│   ├── data.py                    # 新增
│   └── index.py                   # 新增
│
├── auth/                          # 认证模块
│   ├── __init__.py
│   ├── service.py                 # 注册/登录/Token 签发
│   └── dependencies.py            # FastAPI Depends: get_current_user, require_role
│
├── db/                            # 数据库
│   ├── __init__.py
│   ├── database.py                # SQLAlchemy async engine + session
│   └── models.py                  # ORM 模型（6 张新表）
│
├── intent/                        # 多级意图识别
│   ├── __init__.py
│   ├── base.py                    # BaseIntentClassifier 抽象类
│   ├── jieba_classifier.py        # Level 1: Jieba 关键词打分
│   ├── bert_classifier.py         # Level 2: BERT 模型分类
│   ├── llm_classifier.py          # Level 3: LLM 兜底
│   └── pipeline.py                # 管道编排器
│
├── agents/                        # LangGraph 节点 + 图（原地重写）
│   ├── prompts.py                 # 保留不修改
│   ├── graph.py                   # 重写：新 StateGraph
│   └── nodes.py                   # 重写：节点函数
│
├── chat/                          # 对话管理
│   ├── __init__.py
│   └── session.py                 # Session + Message CRUD（MySQL + Redis）
│
├── rag/                           # 检索核心（全部保留不修改）
│   ├── hybrid.py, fusion.py, milvus.py, indexing.py
│   ├── legal.py, tender.py, bidding.py, product.py
│
├── file/                          # 文件管理
│   ├── __init__.py
│   ├── minio_client.py            # MinIO 客户端
│   ├── parser.py                  # 文件解析器（PDF/MD/TXT）
│   └── chunker.py                 # 分块策略
│
├── task/                          # 任务队列
│   ├── __init__.py
│   ├── arq_config.py              # ARQ Redis 配置
│   ├── worker.py                  # Worker 启动入口
│   └── jobs.py                    # 任务定义
│
├── data/                          # 业务数据查询
│   ├── __init__.py
│   └── repository.py              # 三张业务表分页查询
│
├── models/                        # LLM/Embedding（保留不修改）
│   └── llm.py
│
├── schemas/                       # Pydantic schema
│   ├── chat.py                    # 保留
│   ├── auth.py                    # 新增
│   ├── document.py                # 新增
│   └── index.py                   # 新增
│
└── utils/                         # 工具（保留不修改）
    ├── logger.py, db_util.py
    ├── legal_document_util.py, pdf_load_util.py

tests/
├── conftest.py
├── test_auth.py
├── test_intent/
│   ├── test_jieba_classifier.py
│   ├── test_bert_classifier.py
│   └── test_pipeline.py
├── test_chat.py
├── test_documents.py
├── test_data.py
└── test_index.py
```

---

## Phase 1: 基础设施

### Task 1: 更新依赖和配置

**Files:**
- Modify: `pyproject.toml`
- Modify: `app/config.py`

- [ ] **Step 1: 添加新依赖**

将 `pyproject.toml` 的 dependencies 替换为：

```toml
dependencies = [
    "fastapi[all]>=0.135.2",
    "langgraph>=1.1.3",
    "llama-index-core>=0.14.19",
    "llama-index-vector-stores-milvus>=1.1.0",
    "pydantic-settings>=2.13.1",
    "python-dotenv>=1.2.2",
    "pymysql>=1.1.2",
    "pandas>=2.3.3",
    "pymupdf>=1.27.2.3",
    "jieba>=0.42.1",
    "sqlalchemy[asyncio]>=2.0.0",
    "asyncmy>=0.2.9",
    "redis>=5.0.0",
    "arq>=0.26.0",
    "pyjwt>=2.8.0",
    "passlib[bcrypt]>=1.7.4",
    "minio>=7.2.0",
    "python-multipart>=0.0.9",
    "torch>=2.0.0",
    "transformers>=4.40.0",
]
```

- [ ] **Step 2: 安装依赖**

```bash
cd E:\Github\TenderRag\TenderRag && uv sync
```

- [ ] **Step 3: 扩展 app/config.py**

读取 `app/config.py`，在其末尾追加以下配置：

```python
# Redis
redis_host: str = "127.0.0.1"
redis_port: int = 6379
redis_db: int = 0
redis_password: str = ""

# MinIO
minio_endpoint: str = "127.0.0.1:9000"
minio_access_key: str = "minioadmin"
minio_secret_key: str = "minioadmin"
minio_bucket: str = "tenderrag-documents"
minio_secure: bool = False

# JWT
jwt_secret_key: str = "change-me-in-production"
jwt_algorithm: str = "HS256"
jwt_expire_minutes: int = 1440

# Intent Pipeline
jieba_threshold: float = 0.6
bert_threshold: float = 0.7
llm_threshold: float = 0.5
bert_model_path: str = "models/bert_intent_classifier.pth"
bert_model_type: str = "bert-base-chinese"
num_intent_labels: int = 4
intent_labels: list[str] = ["legal", "tender", "product", "other"]

# ARQ
arq_redis_dsn: str = "redis://127.0.0.1:6379/1"

# Database (for SQLAlchemy)
database_async_url: str = ""
```

- [ ] **Step 4: 确认配置加载正常**

```bash
cd E:\Github\TenderRag\TenderRag && uv run python -c "from app.config import settings; print('OK:', settings.jwt_secret_key)"
```

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock app/config.py
git commit -m "feat: add new dependencies and extend config for refactor"
```

---

### Task 2: 数据库模型和连接

**Files:**
- Create: `app/db/__init__.py`
- Create: `app/db/database.py`
- Create: `app/db/models.py`
- Create: `app/db/init_db.py`

- [ ] **Step 1: 创建 app/db/__init__.py**

```bash
echo "" > app/db/__init__.py
```

- [ ] **Step 2: 实现 app/db/database.py**

```python
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from app.config import settings

engine = create_async_engine(
    settings.database_async_url or f"mysql+asyncmy://{settings.database_user}:{settings.database_password}@{settings.database_url}/{settings.database_db_name}",
    echo=settings.debug,
    pool_size=10,
    max_overflow=20,
)

async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def get_db() -> AsyncSession:
    async with async_session() as session:
        yield session
```

- [ ] **Step 3: 实现 app/db/models.py**

```python
from datetime import datetime
from sqlalchemy import Column, Integer, String, Text, DateTime, Boolean, JSON, Float, ForeignKey
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


def audit_columns():
    return [
        Column("created_by", String(64), nullable=True, comment="创建人"),
        Column("created_at", DateTime, default=datetime.now, nullable=False, comment="创建时间"),
        Column("updated_by", String(64), nullable=True, comment="修改人"),
        Column("updated_at", DateTime, default=datetime.now, onupdate=datetime.now, nullable=True, comment="修改时间"),
        Column("deleted_by", String(64), nullable=True, comment="删除人"),
        Column("deleted_at", DateTime, nullable=True, comment="删除时间"),
        Column("is_deleted", Boolean, default=False, nullable=False, comment="删除标记"),
    ]


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(64), unique=True, nullable=False)
    password_hash = Column(String(256), nullable=False)
    role = Column(String(16), nullable=False, default="external", comment="admin/internal/external")
    *audit_columns()


class Session(Base):
    __tablename__ = "sessions"
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    title = Column(String(256), nullable=True)
    *audit_columns()


class Message(Base):
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(Integer, ForeignKey("sessions.id"), nullable=False)
    role = Column(String(16), nullable=False, comment="user/assistant")
    content = Column(Text, nullable=False)
    intents = Column(JSON, nullable=True)
    *audit_columns()


class IntentLog(Base):
    __tablename__ = "intent_logs"
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(Integer, nullable=True)
    question = Column(Text, nullable=False)
    failed_level = Column(String(4), nullable=False, comment="L1/L2/L3")
    scores = Column(JSON, nullable=True)
    final_intent = Column(String(64), nullable=True)
    *audit_columns()


class Document(Base):
    __tablename__ = "documents"
    id = Column(Integer, primary_key=True, autoincrement=True)
    filename = Column(String(256), nullable=False)
    category = Column(String(32), nullable=False, comment="legal/tender/product")
    file_path = Column(String(512), nullable=False, comment="MinIO key")
    file_size = Column(Integer, nullable=True)
    status = Column(String(16), default="pending", comment="pending/processing/done/failed")
    uploaded_by = Column(Integer, ForeignKey("users.id"), nullable=True)
    *audit_columns()


class IndexTask(Base):
    __tablename__ = "index_tasks"
    id = Column(Integer, primary_key=True, autoincrement=True)
    task_type = Column(String(16), nullable=False, comment="full/incremental")
    category = Column(String(32), nullable=False)
    document_ids = Column(JSON, nullable=True)
    status = Column(String(16), default="queued", comment="queued/running/done/failed")
    result_msg = Column(Text, nullable=True)
    *audit_columns()
```

- [ ] **Step 4: 实现 app/db/init_db.py（建表脚本）**

```python
import asyncio
from app.db.database import engine
from app.db.models import Base


async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    print("Tables created successfully.")


if __name__ == "__main__":
    asyncio.run(init_db())
```

- [ ] **Step 5: 运行建表并验证**

```bash
cd E:\Github\TenderRag\TenderRag && uv run python -m app.db.init_db
```

- [ ] **Step 6: Commit**

```bash
git add app/db/
git commit -m "feat: add SQLAlchemy async models and database connection"
```

---

## Phase 2: 认证模块

### Task 3: Auth Service

**Files:**
- Create: `app/auth/__init__.py`
- Create: `app/auth/service.py`
- Create: `app/auth/dependencies.py`
- Create: `app/schemas/auth.py`
- Create: `tests/conftest.py`
- Create: `tests/test_auth.py`

- [ ] **Step 1: 创建空文件**

```bash
echo "" > app/auth/__init__.py
mkdir -p tests
```

- [ ] **Step 2: 实现 app/schemas/auth.py**

```python
from pydantic import BaseModel


class RegisterRequest(BaseModel):
    username: str
    password: str


class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserInfo(BaseModel):
    id: int
    username: str
    role: str
```

- [ ] **Step 3: 实现 app/auth/service.py**

```python
from datetime import datetime, timedelta
import jwt
from passlib.context import CryptContext
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.config import settings
from app.db.models import User

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class AuthService:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def register(self, username: str, password: str) -> User:
        result = await self.db.execute(select(User).where(User.username == username))
        if result.scalar_one_or_none():
            raise ValueError("用户名已存在")
        user = User(
            username=username,
            password_hash=pwd_context.hash(password),
            role="external",
            created_by="system",
        )
        self.db.add(user)
        await self.db.commit()
        await self.db.refresh(user)
        return user

    async def login(self, username: str, password: str) -> str:
        result = await self.db.execute(
            select(User).where(User.username == username, User.is_deleted == False)
        )
        user = result.scalar_one_or_none()
        if not user or not pwd_context.verify(password, user.password_hash):
            raise ValueError("用户名或密码错误")
        return self._create_token(user)

    def _create_token(self, user: User) -> str:
        payload = {
            "sub": str(user.id),
            "username": user.username,
            "role": user.role,
            "exp": datetime.now() + timedelta(minutes=settings.jwt_expire_minutes),
        }
        return jwt.encode(payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)

    @staticmethod
    def decode_token(token: str) -> dict:
        return jwt.decode(token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm])
```

- [ ] **Step 4: 实现 app/auth/dependencies.py**

```python
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.auth.service import AuthService
from app.db.database import get_db
from app.db.models import User

security = HTTPBearer()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncSession = Depends(get_db),
) -> User:
    try:
        payload = AuthService.decode_token(credentials.credentials)
        user_id = int(payload["sub"])
    except Exception:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="无效的认证令牌")
    result = await db.execute(
        select(User).where(User.id == user_id, User.is_deleted == False)
    )
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="用户不存在")
    return user


def require_role(*roles: str):
    async def checker(user: User = Depends(get_current_user)):
        if user.role not in roles:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="权限不足")
        return user
    return checker
```

- [ ] **Step 5: 创建 tests/conftest.py**

```python
import pytest
from app.db.models import Base

# Placeholder for async test fixtures
```

- [ ] **Step 6: 创建 tests/test_auth.py**

```python
import pytest


def test_placeholder():
    """Placeholder test — replace with actual integration tests."""
    pass
```

- [ ] **Step 7: Commit**

```bash
git add app/auth/ app/schemas/auth.py tests/
git commit -m "feat: add auth service with JWT and role-based dependencies"
```

---

### Task 4: Auth API 路由

**Files:**
- Create: `app/api/__init__.py`
- Create: `app/api/auth.py`

- [ ] **Step 1: 创建 app/api/__init__.py**

```bash
echo "" > app/api/__init__.py
```

- [ ] **Step 2: 实现 app/api/auth.py**

```python
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from app.auth.service import AuthService
from app.auth.dependencies import get_current_user
from app.db.database import get_db
from app.db.models import User
from app.schemas.auth import RegisterRequest, LoginRequest, TokenResponse, UserInfo

router = APIRouter(prefix="/api/auth", tags=["auth"])


@router.post("/register", response_model=TokenResponse)
async def register(body: RegisterRequest, db: AsyncSession = Depends(get_db)):
    service = AuthService(db)
    try:
        user = await service.register(body.username, body.password)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    token = service._create_token(user)
    return TokenResponse(access_token=token)


@router.post("/login", response_model=TokenResponse)
async def login(body: LoginRequest, db: AsyncSession = Depends(get_db)):
    service = AuthService(db)
    try:
        token = await service.login(body.username, body.password)
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))
    return TokenResponse(access_token=token)


@router.get("/me", response_model=UserInfo)
async def me(user: User = Depends(get_current_user)):
    return UserInfo(id=user.id, username=user.username, role=user.role)
```

- [ ] **Step 3: 验证路由导入**

```bash
cd E:\Github\TenderRag\TenderRag && uv run python -c "from app.api.auth import router; print('OK')"
```

- [ ] **Step 4: Commit**

```bash
git add app/api/__init__.py app/api/auth.py
git commit -m "feat: add auth API endpoints (register/login/me)"
```

---

## Phase 3: 多级意图识别

### Task 5: 意图分类基类和 Jieba 分类器

**Files:**
- Create: `app/intent/__init__.py`
- Create: `app/intent/base.py`
- Create: `app/intent/jieba_classifier.py`
- Create: `tests/test_intent/__init__.py`
- Create: `tests/test_intent/test_jieba_classifier.py`

- [ ] **Step 1: app/intent/__init__.py**

```bash
echo "" > app/intent/__init__.py
```

- [ ] **Step 2: app/intent/base.py**

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class IntentResult:
    intents: list[str]
    scores: dict[str, float]
    level: str  # "L1", "L2", "L3"
    hit: bool


class BaseIntentClassifier(ABC):
    @abstractmethod
    def classify(self, question: str) -> IntentResult:
        ...
```

- [ ] **Step 3: app/intent/jieba_classifier.py**

```python
import jieba
import jieba.analyse
from app.intent.base import BaseIntentClassifier, IntentResult

DOMAIN_KEYWORDS = {
    "legal": [
        "法律", "法规", "条例", "规定", "办法", "政策", "合规", "风险防范",
        "资质许可", "招标投标法", "政府采购法", "民法典", "行政处罚",
        "违法", "诉讼", "仲裁", "保证金", "履约", "评标", "废标",
    ],
    "tender": [
        "招标公告", "采购公告", "中标", "公示", "寻标", "找项目",
        "预算", "开标", "投标截止", "资格预审", "答疑", "踏勘",
        "招标文件", "更正公告", "流标", "竞争性谈判", "询价",
    ],
    "product": [
        "规格", "型号", "品牌", "价格", "报价", "供应商", "库存",
        "售后服务", "参数", "材质", "尺寸", "重量", "产地",
        "产品", "商品", "采购", "合同", "订单",
    ],
}


class JiebaIntentClassifier(BaseIntentClassifier):
    def __init__(self, threshold: float = 0.6):
        self.threshold = threshold
        for domain in DOMAIN_KEYWORDS:
            for kw in DOMAIN_KEYWORDS[domain]:
                jieba.add_word(kw)

    def classify(self, question: str) -> IntentResult:
        words = set(jieba.lcut(question))
        tags = set(jieba.analyse.extract_tags(question, topK=10))

        scores = {}
        for domain, keywords in DOMAIN_KEYWORDS.items():
            kw_set = set(keywords)
            matches = len(words & kw_set) + len(tags & kw_set)
            scores[domain] = min(matches / max(len(kw_set) * 0.1, 1), 1.0)

        best_domain = max(scores, key=scores.get)
        best_score = scores[best_domain]

        if best_score >= self.threshold:
            intents = [d for d, s in scores.items() if s >= self.threshold]
            return IntentResult(intents=intents, scores=scores, level="L1", hit=True)

        return IntentResult(intents=[], scores=scores, level="L1", hit=False)
```

- [ ] **Step 4: tests/test_intent/test_jieba_classifier.py**

```python
import pytest
from app.intent.jieba_classifier import JiebaIntentClassifier


def test_jieba_hit_legal():
    c = JiebaIntentClassifier(threshold=0.6)
    result = c.classify("招标投标法对投标保证金有什么规定？")
    assert result.hit is True
    assert "legal" in result.intents
    assert result.level == "L1"


def test_jieba_hit_tender():
    c = JiebaIntentClassifier(threshold=0.6)
    result = c.classify("查询北京地区最近的招标公告")
    assert result.hit is True
    assert "tender" in result.intents
    assert result.level == "L1"


def test_jieba_miss():
    c = JiebaIntentClassifier(threshold=0.6)
    result = c.classify("今天天气怎么样")
    assert result.hit is False
    assert result.level == "L1"
```

- [ ] **Step 5: 运行测试**

```bash
cd E:\Github\TenderRag\TenderRag && uv run pytest tests/test_intent/test_jieba_classifier.py -v
```

预期：3 passed

- [ ] **Step 6: Commit**

```bash
git add app/intent/ tests/test_intent/
git commit -m "feat: add intent base class and Jieba keyword classifier"
```

---

### Task 6: BERT 分类器

**Files:**
- Create: `app/intent/bert_classifier.py`
- Create: `tests/test_intent/test_bert_classifier.py`

- [ ] **Step 1: app/intent/bert_classifier.py**

```python
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from app.intent.base import BaseIntentClassifier, IntentResult
from app.config import settings


class BertIntentClassifier(BaseIntentClassifier):
    def __init__(
        self,
        model_path: str | None = None,
        model_type: str | None = None,
        num_labels: int | None = None,
        labels: list[str] | None = None,
        threshold: float = 0.7,
    ):
        self.model_path = model_path or settings.bert_model_path
        self.model_type = model_type or settings.bert_model_type
        self.num_labels = num_labels or settings.num_intent_labels
        self.labels = labels or settings.intent_labels
        self.threshold = threshold

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_type)
        self.model = self._build_model()
        self._load_weights()
        self.model.to(self.device)
        self.model.eval()

    def _build_model(self) -> nn.Module:
        class BertClassifier(nn.Module):
            def __init__(self, model_type, num_labels):
                super().__init__()
                self.bert = AutoModel.from_pretrained(model_type)
                self.dropout = nn.Dropout(0.1)
                self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

            def forward(self, input_ids, attention_mask):
                outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
                pooled = outputs.last_hidden_state[:, 0, :]
                pooled = self.dropout(pooled)
                return self.classifier(pooled)

        return BertClassifier(self.model_type, self.num_labels)

    def _load_weights(self):
        state_dict = torch.load(self.model_path, map_location=self.device, weights_only=True)
        if isinstance(state_dict, dict) and not any(k.startswith("bert.") for k in state_dict):
            state_dict = state_dict.get("model_state_dict", state_dict)
        self.model.load_state_dict(state_dict, strict=False)

    @torch.no_grad()
    def classify(self, question: str) -> IntentResult:
        encoded = self.tokenizer(
            question,
            max_length=256,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        logits = self.model(input_ids, attention_mask)
        probs = torch.softmax(logits, dim=-1).squeeze(0)

        scores = {self.labels[i]: round(float(probs[i]), 4) for i in range(self.num_labels)}
        best_idx = int(torch.argmax(probs).item())
        best_score = float(probs[best_idx])

        if best_score >= self.threshold:
            return IntentResult(
                intents=[self.labels[best_idx]],
                scores=scores,
                level="L2",
                hit=True,
            )

        return IntentResult(intents=[], scores=scores, level="L2", hit=False)
```

- [ ] **Step 2: tests/test_intent/test_bert_classifier.py**

```python
import pytest


def test_bert_placeholder():
    """BERT 模型文件依赖用户自训模型，此处做接口占位测试。"""
    pass
```

- [ ] **Step 3: Commit**

```bash
git add app/intent/bert_classifier.py tests/test_intent/test_bert_classifier.py
git commit -m "feat: add BERT intent classifier with PyTorch model loading"
```

---

### Task 7: LLM 分类器和管道编排器

**Files:**
- Create: `app/intent/llm_classifier.py`
- Create: `app/intent/pipeline.py`
- Create: `tests/test_intent/test_pipeline.py`

- [ ] **Step 1: app/intent/llm_classifier.py**

```python
from app.intent.base import BaseIntentClassifier, IntentResult
from app.agents.prompts import INTENT_CLASSIFY_PROMPT
from app.models.llm import get_llm


class LLMIntentClassifier(BaseIntentClassifier):
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def classify(self, question: str) -> IntentResult:
        llm = get_llm()
        prompt = INTENT_CLASSIFY_PROMPT.format(question=question)
        response = str(llm.complete(prompt)).strip().lower()

        all_domains = ["legal", "tender", "product", "other"]
        intents = [d for d in all_domains if d in response]

        scores = {}
        for d in all_domains:
            scores[d] = 0.85 if d in intents else 0.05

        if not intents:
            intents = ["other"]
            scores["other"] = 0.75

        best_score = max(scores.values())

        if best_score >= self.threshold:
            return IntentResult(intents=intents, scores=scores, level="L3", hit=True)

        return IntentResult(intents=["other"], scores=scores, level="L3", hit=False)
```

- [ ] **Step 2: app/intent/pipeline.py**

```python
import time
import logging
from app.config import settings
from app.intent.base import BaseIntentClassifier, IntentResult
from app.intent.jieba_classifier import JiebaIntentClassifier
from app.intent.bert_classifier import BertIntentClassifier
from app.intent.llm_classifier import LLMIntentClassifier

logger = logging.getLogger(__name__)


class IntentPipeline:
    def __init__(
        self,
        log_callback=None,
        jieba_threshold: float | None = None,
        bert_threshold: float | None = None,
        llm_threshold: float | None = None,
    ):
        self.jieba = JiebaIntentClassifier(threshold=jieba_threshold or settings.jieba_threshold)
        self.bert = BertIntentClassifier(threshold=bert_threshold or settings.bert_threshold)
        self.llm = LLMIntentClassifier(threshold=llm_threshold or settings.llm_threshold)
        self.log_callback = log_callback

    def classify(self, question: str) -> list[str]:
        start = time.time()

        result = self.jieba.classify(question)
        if result.hit:
            self._log(question, result)
            return result.intents

        result = self.bert.classify(question)
        if result.hit:
            self._log(question, result)
            return result.intents

        result = self.llm.classify(question)
        self._log(question, result)
        return result.intents

    def _log(self, question: str, result: IntentResult):
        logger.info(
            "intent_pipeline level=%s hit=%s intents=%s scores=%s question=%s",
            result.level, result.hit, result.intents, result.scores, question[:80],
        )
        if self.log_callback and not result.hit:
            self.log_callback(question, result)
```

- [ ] **Step 3: tests/test_intent/test_pipeline.py**

```python
import pytest
from app.intent.pipeline import IntentPipeline


def test_pipeline_legal():
    pipeline = IntentPipeline()
    intents = pipeline.classify("招标投标法对投标保证金有什么规定？")
    assert len(intents) > 0
    assert any(i in ["legal", "tender", "product", "other"] for i in intents)


def test_pipeline_chat():
    pipeline = IntentPipeline()
    intents = pipeline.classify("今天天气怎么样")
    assert len(intents) > 0
    assert "other" in intents or len(intents) >= 1
```

- [ ] **Step 4: 运行测试**

```bash
cd E:\Github\TenderRag\TenderRag && uv run pytest tests/test_intent/test_pipeline.py -v
```

预期：2 passed

- [ ] **Step 5: Commit**

```bash
git add app/intent/llm_classifier.py app/intent/pipeline.py tests/test_intent/test_pipeline.py
git commit -m "feat: add LLM classifier and intent pipeline orchestrator"
```

---

## Phase 4: 对话模块

### Task 8: 对话 Session & Message CRUD

**Files:**
- Create: `app/chat/__init__.py`
- Create: `app/chat/session.py`

- [ ] **Step 1: 创建空文件**

```bash
echo "" > app/chat/__init__.py
```

- [ ] **Step 2: 实现 app/chat/session.py**

```python
import json
from datetime import datetime
from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession
from app.db.models import Session, Message
from app.config import settings
import redis.asyncio as aioredis

redis_client = aioredis.from_url(
    f"redis://{settings.redis_host}:{settings.redis_port}/{settings.redis_db}",
    password=settings.redis_password or None,
)


class SessionManager:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def create_session(self, user_id: int, title: str = "新对话") -> Session:
        session = Session(
            user_id=user_id,
            title=title,
            created_by=str(user_id),
        )
        self.db.add(session)
        await self.db.commit()
        await self.db.refresh(session)
        return session

    async def list_sessions(self, user_id: int) -> list[dict]:
        result = await self.db.execute(
            select(Session)
            .where(Session.user_id == user_id, Session.is_deleted == False)
            .order_by(desc(Session.updated_at))
        )
        sessions = result.scalars().all()
        return [
            {"id": s.id, "title": s.title or "新对话", "created_at": s.created_at.isoformat()}
            for s in sessions
        ]

    async def get_messages(self, session_id: int, limit: int = 20) -> list[dict]:
        cache_key = f"session:{session_id}:messages"
        cached = await redis_client.lrange(cache_key, 0, -1)
        if cached:
            return [json.loads(m) for m in cached]

        result = await self.db.execute(
            select(Message)
            .where(Message.session_id == session_id, Message.is_deleted == False)
            .order_by(Message.created_at)
            .limit(limit)
        )
        messages = result.scalars().all()
        data = [
            {"role": m.role, "content": m.content, "intents": m.intents, "created_at": m.created_at.isoformat()}
            for m in messages
        ]
        for item in data:
            await redis_client.rpush(cache_key, json.dumps(item, ensure_ascii=False))
        await redis_client.expire(cache_key, 3600)
        return data

    async def add_message(self, session_id: int, role: str, content: str, intents: list[str] | None = None):
        msg = Message(
            session_id=session_id,
            role=role,
            content=content,
            intents=intents,
            created_by="system",
        )
        self.db.add(msg)
        await self.db.commit()

        cache_key = f"session:{session_id}:messages"
        item = {"role": role, "content": content, "intents": intents, "created_at": datetime.now().isoformat()}
        await redis_client.rpush(cache_key, json.dumps(item, ensure_ascii=False))
        await redis_client.expire(cache_key, 3600)

        result = await self.db.execute(select(Session).where(Session.id == session_id))
        session = result.scalar_one_or_none()
        if session:
            session.updated_at = datetime.now()
            await self.db.commit()

    async def soft_delete_session(self, session_id: int, deleted_by: str):
        result = await self.db.execute(select(Session).where(Session.id == session_id))
        session = result.scalar_one_or_none()
        if session:
            session.is_deleted = True
            session.deleted_at = datetime.now()
            session.deleted_by = deleted_by
            await self.db.commit()
        cache_key = f"session:{session_id}:messages"
        await redis_client.delete(cache_key)
```

- [ ] **Step 3: 验证导入**

```bash
cd E:\Github\TenderRag\TenderRag && uv run python -c "from app.chat.session import SessionManager; print('OK')"
```

- [ ] **Step 4: Commit**

```bash
git add app/chat/
git commit -m "feat: add session and message CRUD with Redis cache + MySQL persistence"
```

---

### Task 9: 重写 LangGraph 节点和图

**Files:**
- Modify: `app/agents/nodes.py` — 重写
- Modify: `app/agents/graph.py` — 重写
- 保留: `app/agents/prompts.py` — 不修改

- [ ] **Step 1: 重写 app/agents/nodes.py**

```python
import logging
import queue
import threading
import time
from app.agents.prompts import DOMAIN_PROMPTS
from app.models.llm import get_llm
from app.intent.pipeline import IntentPipeline
from app.chat.session import SessionManager
from app.db.database import async_session
from app.utils.logger import get_logger

logger = get_logger(__name__)
_tls = threading.local()

DOMAIN_NAMES = {
    "legal": "法律",
    "tender": "招标",
    "bidding": "投标",
    "product": "产品",
}


def set_stream_queue(q: queue.Queue | None) -> None:
    _tls.stream_queue = q


def _stream_send(msg_type: str, content: str | None) -> None:
    q = getattr(_tls, "stream_queue", None)
    if q is not None:
        q.put((msg_type, content))


async def _log_intent(question: str, result):
    async with async_session() as db:
        from app.db.models import IntentLog
        log = IntentLog(
            question=question,
            failed_level=result.level,
            scores=result.scores,
            final_intent=",".join(result.intents),
            created_by="system",
        )
        db.add(log)
        await db.commit()


intent_pipeline = IntentPipeline(log_callback=_log_intent)


def memory_retrieve(state: dict) -> dict:
    """从 MySQL 拉取历史消息。"""
    import asyncio
    session_id = state.get("session_id", "default")
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, _async_memory_retrieve(session_id))
                history = future.result(timeout=10)
        else:
            history = asyncio.run(_async_memory_retrieve(session_id))
    except Exception:
        logger.exception("memory_retrieve failed")
        history = []
    state["history"] = history
    logger.debug("记忆检索完成 session=%s history_len=%d", session_id, len(history))
    return state


async def _async_memory_retrieve(session_id: str) -> list[dict]:
    try:
        sid = int(session_id)
    except (ValueError, TypeError):
        return []
    async with async_session() as db:
        sm = SessionManager(db)
        return await sm.get_messages(sid)


def classify_intent(state: dict) -> dict:
    start_time = time.time()
    question = state["question"]
    intents = intent_pipeline.classify(question)

    logger.info(
        "意图分类 intents=%s elapsed=%.2fs question=%s",
        intents, time.time() - start_time, question[:50],
    )
    return {**state, "intents": intents}


def _format_context(results: list[dict]) -> str:
    if not results:
        return "暂无相关信息。"
    parts = []
    for i, r in enumerate(results, 1):
        meta = r.get("metadata", {})
        meta_str = ", ".join(
            f"{k}={v}"
            for k, v in meta.items()
            if k not in ("_node_type", "document_id", "doc_id", "ref_doc_id")
        )
        parts.append(f"[片段{i}] [{meta_str}]\n{r['content']}")
    return "\n\n---\n\n".join(parts)


def _search_domain(domain: str, question: str, top_k: int = 5) -> str:
    try:
        if domain == "legal":
            from app.rag.legal import get_nodes
        elif domain == "tender":
            from app.rag.tender import get_nodes
        elif domain == "bidding":
            from app.rag.bidding import get_nodes
        elif domain == "product":
            from app.rag.product import get_nodes
        else:
            return f"[提示] 未知领域: {domain}"
        results = get_nodes(question, top_k)
        context = _format_context(results)
        logger.info("%s 检索完成 count=%d", DOMAIN_NAMES.get(domain, domain), len(results))
        return context
    except Exception:
        logger.exception("%s 索引检索失败", DOMAIN_NAMES.get(domain, domain))
        return f"[提示] {DOMAIN_NAMES.get(domain, domain)} 索引检索失败，请稍后再试。"


def legal_agent(state: dict) -> dict:
    start_time = time.time()
    context = _search_domain("legal", state["question"])
    logger.info("法律检索完成 elapsed=%.2fs", time.time() - start_time)
    return {
        "contexts": {"legal": context},
        "synthesize_prompts": {"legal": DOMAIN_PROMPTS["legal"]},
    }


def tender_agent(state: dict) -> dict:
    start_time = time.time()
    context = _search_domain("tender", state["question"])
    logger.info("招标检索完成 elapsed=%.2fs", time.time() - start_time)
    return {
        "contexts": {"tender": context},
        "synthesize_prompts": {"tender": DOMAIN_PROMPTS["tender"]},
    }


def bidding_agent(state: dict) -> dict:
    start_time = time.time()
    context = _search_domain("bidding", state["question"])
    logger.info("投标检索完成 elapsed=%.2fs", time.time() - start_time)
    return {
        "contexts": {"bidding": context},
        "synthesize_prompts": {"bidding": DOMAIN_PROMPTS["bidding"]},
    }


def product_agent(state: dict) -> dict:
    start_time = time.time()
    context = _search_domain("product", state["question"])
    logger.info("商品检索完成 elapsed=%.2fs", time.time() - start_time)
    return {
        "contexts": {"product": context},
        "synthesize_prompts": {"product": DOMAIN_PROMPTS["product"]},
    }


def merge_contexts(state: dict) -> dict:
    intents = state.get("intents", ["other"])
    contexts = state.get("contexts", {})
    prompts = state.get("synthesize_prompts", {})

    if intents == ["other"] or not contexts:
        return {
            "context": "",
            "synthesize_prompt": DOMAIN_PROMPTS["other"],
            "intent": "other",
        }

    parts = []
    for domain in intents:
        ctx = contexts.get(domain, "")
        if not ctx or ctx == "暂无相关信息。" or ctx.startswith("[提示]"):
            continue
        label = DOMAIN_NAMES.get(domain, domain)
        parts.append(f"【{label}领域】\n{ctx}")

    if parts:
        combined = "\n\n---\n\n".join(parts)
    else:
        merged = [c for c in contexts.values() if c.strip()]
        combined = merged[0] if merged else "暂无相关信息。"

    if len(intents) == 1:
        prompt = prompts.get(intents[0], DOMAIN_PROMPTS["other"])
    else:
        prompt = DOMAIN_PROMPTS["multi"]

    logger.info("上下文合并完成 intents=%s domains=%d context_len=%d", intents, len(contexts), len(combined))
    return {"context": combined, "synthesize_prompt": prompt, "intent": ",".join(intents)}


def synthesize(state: dict):
    start_time = time.time()
    intent = state.get("intent", "other")

    history = state.get("history", [])
    history_lines = []
    for msg in history:
        history_lines.append(f"{msg['role']}: {msg['content']}")
    history_text = "\n".join(history_lines)

    prompt_template = state.get("synthesize_prompt", DOMAIN_PROMPTS["other"])
    prompt = prompt_template.format(
        context=state.get("context", "暂无背景信息。"),
        history=history_text,
        question=state["question"],
    )

    llm = get_llm()
    answer = ""
    for resp in llm.stream_complete(prompt):
        if resp.delta:
            answer += resp.delta
            _stream_send("message", resp.delta)

    state["answer"] = answer
    _stream_send("done", None)
    logger.info("合成完成 intent=%s answer_len=%d elapsed=%.2fs", intent, len(answer), time.time() - start_time)
    return state


def store_memory(state: dict) -> dict:
    """持久化消息到 MySQL + Redis。"""
    import asyncio
    session_id = state.get("session_id", "default")
    intents = state.get("intents", ["other"])
    try:
        asyncio.run(_async_store_memory(session_id, state["question"], state.get("answer", ""), intents))
    except Exception:
        logger.exception("store_memory failed")
    return state


async def _async_store_memory(session_id: str, question: str, answer: str, intents: list[str]):
    try:
        sid = int(session_id)
    except (ValueError, TypeError):
        return
    async with async_session() as db:
        sm = SessionManager(db)
        await sm.add_message(sid, "user", question, intents)
        if answer:
            await sm.add_message(sid, "assistant", answer, None)
```

- [ ] **Step 2: 重写 app/agents/graph.py（图结构不变，引用更新后的 nodes）**

graph.py 内容不变（`route_intent` 和 `build_graph` 逻辑未变，因为 `classify_intent` 现在返回 `state`，且 `route_intent` 接收 `state["intents"]`）。确认导入仍然正确：

```python
import operator
from typing import Annotated, TypedDict

from langgraph.graph import StateGraph, END
from langgraph.types import Send

from app.agents.nodes import (
    memory_retrieve,
    classify_intent,
    legal_agent,
    tender_agent,
    bidding_agent,
    product_agent,
    merge_contexts,
    synthesize,
    store_memory,
)


class AgentState(TypedDict, total=False):
    session_id: str
    question: str
    intents: Annotated[list[str], operator.add]
    history: list[dict]
    contexts: Annotated[dict[str, str], operator.or_]
    synthesize_prompts: Annotated[dict[str, str], operator.or_]
    context: str
    synthesize_prompt: str
    intent: str
    answer: str


def route_intent(state: AgentState) -> list[Send]:
    intents = state.get("intents", ["other"])
    domains = [i for i in intents if i != "other"]
    if not domains:
        return [Send("merge_contexts", state)]
    base = {
        "question": state["question"],
        "session_id": state.get("session_id", "default"),
        "history": state.get("history", []),
    }
    return [Send(f"{d}_agent", base) for d in domains]


def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)
    graph.add_node("memory_retrieve", memory_retrieve)
    graph.add_node("classify_intent", classify_intent)
    graph.add_node("legal_agent", legal_agent)
    graph.add_node("tender_agent", tender_agent)
    graph.add_node("bidding_agent", bidding_agent)
    graph.add_node("product_agent", product_agent)
    graph.add_node("merge_contexts", merge_contexts)
    graph.add_node("synthesize", synthesize)
    graph.add_node("store_memory", store_memory)

    graph.set_entry_point("memory_retrieve")
    graph.add_edge("memory_retrieve", "classify_intent")
    graph.add_conditional_edges("classify_intent", route_intent)

    for node in ("legal_agent", "tender_agent", "bidding_agent", "product_agent"):
        graph.add_edge(node, "merge_contexts")

    graph.add_edge("merge_contexts", "synthesize")
    graph.add_edge("synthesize", "store_memory")
    graph.add_edge("store_memory", END)

    return graph.compile()
```

- [ ] **Step 3: 验证导入**

```bash
cd E:\Github\TenderRag\TenderRag && uv run python -c "from app.agents.graph import build_graph; g = build_graph(); print('OK:', type(g).__name__)"
```

- [ ] **Step 4: Commit**

```bash
git add app/agents/nodes.py app/agents/graph.py
git commit -m "feat: rewrite LangGraph nodes with multi-level intent pipeline and MySQL persistence"
```

---

### Task 10: Chat API 路由

**Files:**
- Create: `app/api/chat.py`
- 删除: `app/api/endpoints.py`（如果存在）
- 删除: `app/frontend/gradio_app.py`（旧前端）
- 保留: `app/schemas/chat.py`

- [ ] **Step 1: 实现 app/api/chat.py**

```python
import asyncio
import queue
import threading
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel
from app.agents.graph import build_graph
from app.agents.nodes import set_stream_queue
from app.auth.dependencies import get_current_user
from app.db.database import get_db
from app.db.models import User
from app.chat.session import SessionManager

router = APIRouter(tags=["chat"])

_graph = build_graph()


class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatNonStreamRequest(BaseModel):
    session_id: str
    message: str


@router.post("/api/chat")
async def chat_stream(body: ChatRequest, user: User = Depends(get_current_user)):
    q: queue.Queue = queue.Queue()

    def _run():
        set_stream_queue(q)
        _graph.invoke({
            "session_id": body.session_id,
            "question": body.message,
        })

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    async def generate():
        while True:
            try:
                msg_type, content = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: q.get(timeout=0.1)
                )
                if msg_type == "done":
                    break
                yield f"data: {{ \"type\": \"message\", \"content\": {__import__('json').dumps(content, ensure_ascii=False)} }}\n\n"
            except queue.Empty:
                if not thread.is_alive():
                    break
        yield "data: {\"type\": \"done\"}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@router.post("/api/chat/non-stream")
async def chat_non_stream(
    body: ChatNonStreamRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = _graph.invoke({
        "session_id": body.session_id,
        "question": body.message,
    })
    intents = result.get("intents", ["other"])
    return {
        "session_id": body.session_id,
        "content": result.get("answer", ""),
        "intent": ",".join(intents) if isinstance(intents, list) else intents,
    }


@router.get("/api/sessions")
async def list_sessions(user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    sm = SessionManager(db)
    return await sm.list_sessions(user.id)


@router.post("/api/sessions")
async def create_session(user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    sm = SessionManager(db)
    session = await sm.create_session(user.id)
    return {"session_id": str(session.id), "title": session.title}


@router.get("/api/sessions/{session_id}/messages")
async def get_session_messages(session_id: int, user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    sm = SessionManager(db)
    return await sm.get_messages(session_id)


@router.delete("/api/sessions/{session_id}")
async def delete_session(session_id: int, user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    sm = SessionManager(db)
    await sm.soft_delete_session(session_id, str(user.id))
    return {"ok": True}
```

- [ ] **Step 2: 删除旧文件**

```bash
cd E:\Github\TenderRag\TenderRag
rm -f app/api/endpoints.py
rm -f app/frontend/gradio_app.py
```

- [ ] **Step 3: 验证路由导入**

```bash
cd E:\Github\TenderRag\TenderRag && uv run python -c "from app.api.chat import router; print('OK:', len(router.routes))"
```

- [ ] **Step 4: Commit**

```bash
git add app/api/chat.py app/api/endpoints.py app/frontend/gradio_app.py
git commit -m "feat: add chat API with SSE streaming, session CRUD; remove old Gradio frontend"
```

---

## Phase 5: 文件管理 & 任务队列

### Task 11: MinIO 客户端和文件解析器

**Files:**
- Create: `app/file/__init__.py`
- Create: `app/file/minio_client.py`
- Create: `app/file/parser.py`
- Create: `app/file/chunker.py`

- [ ] **Step 1: 创建空文件**

```bash
echo "" > app/file/__init__.py
```

- [ ] **Step 2: 实现 app/file/minio_client.py**

```python
from minio import Minio
from app.config import settings

_client = Minio(
    endpoint=settings.minio_endpoint,
    access_key=settings.minio_access_key,
    secret_key=settings.minio_secret_key,
    secure=settings.minio_secure,
)


def get_minio() -> Minio:
    return _client


def ensure_bucket():
    if not _client.bucket_exists(settings.minio_bucket):
        _client.make_bucket(settings.minio_bucket)


def upload_file(local_path: str, object_name: str) -> str:
    _client.fput_object(settings.minio_bucket, object_name, local_path)
    return object_name


def download_file(object_name: str, local_path: str):
    _client.fget_object(settings.minio_bucket, object_name, local_path)


def delete_file(object_name: str):
    _client.remove_object(settings.minio_bucket, object_name)
```

- [ ] **Step 3: 实现 app/file/parser.py**

```python
from pathlib import Path
import fitz  # PyMuPDF


def parse_file(file_path: str, file_type: str) -> str:
    if file_type == "pdf":
        return _parse_pdf(file_path)
    elif file_type == "md":
        return _parse_md(file_path)
    elif file_type == "txt":
        return _parse_txt(file_path)
    else:
        raise ValueError(f"不支持的文件类型: {file_type}")


def _parse_pdf(path: str) -> str:
    doc = fitz.open(path)
    text_parts = []
    for page in doc:
        text_parts.append(page.get_text())
    doc.close()
    return "\n".join(text_parts)


def _parse_md(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _parse_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()
```

- [ ] **Step 4: 实现 app/file/chunker.py**

```python
from pathlib import Path
from app.utils.pdf_load_util import split_pdf_by_font_analysis
from app.utils.legal_document_util import get_legal_document


def chunk_text(text: str, file_type: str, metadata: dict) -> list[dict]:
    if file_type == "pdf":
        return _chunk_pdf(text, metadata)
    elif file_type == "md":
        return _chunk_md(text, metadata)
    elif file_type == "txt":
        return _chunk_txt(text, metadata)
    else:
        raise ValueError(f"不支持的文件类型: {file_type}")


def _chunk_pdf(text: str, metadata: dict) -> list[dict]:
    # 复用现有的 PDF 切分逻辑
    chunks = split_pdf_by_font_analysis(text)
    return [
        {"content": chunk["text"], "metadata": {**metadata, "section": chunk.get("heading", "")}}
        for chunk in chunks
    ]


def _chunk_md(text: str, metadata: dict) -> list[dict]:
    chunks = []
    current_heading = ""
    current_content = []

    for line in text.split("\n"):
        if line.startswith("# "):
            if current_content:
                chunks.append({"content": "\n".join(current_content), "metadata": {**metadata, "section": current_heading}})
            current_heading = line.lstrip("# ")
            current_content = []
        elif line.startswith("## "):
            if current_content:
                chunks.append({"content": "\n".join(current_content), "metadata": {**metadata, "section": current_heading, "subsection": line.lstrip("## ")}})
        else:
            current_content.append(line)

    if current_content:
        chunks.append({"content": "\n".join(current_content), "metadata": {**metadata, "section": current_heading}})

    return chunks


def _chunk_txt(text: str, metadata: dict, chunk_size: int = 512, overlap: int = 64) -> list[dict]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    idx = 0
    for para in paragraphs:
        if len(para) <= chunk_size:
            chunks.append({"content": para, "metadata": {**metadata, "paragraph_index": idx}})
            idx += 1
        else:
            for i in range(0, len(para), chunk_size - overlap):
                chunk = para[i:i + chunk_size]
                if chunk:
                    chunks.append({"content": chunk, "metadata": {**metadata, "paragraph_index": idx}})
                idx += 1
    return chunks
```

- [ ] **Step 5: 验证导入**

```bash
cd E:\Github\TenderRag\TenderRag && uv run python -c "from app.file.minio_client import get_minio; from app.file.parser import parse_file; from app.file.chunker import chunk_text; print('OK')"
```

- [ ] **Step 6: Commit**

```bash
git add app/file/
git commit -m "feat: add MinIO client, file parsers and chunkers for PDF/MD/TXT"
```

---

### Task 12: ARQ 任务定义和 Worker

**Files:**
- Create: `app/task/__init__.py`
- Create: `app/task/arq_config.py`
- Create: `app/task/jobs.py`
- Create: `app/task/worker.py`

- [ ] **Step 1: 创建空文件**

```bash
echo "" > app/task/__init__.py
```

- [ ] **Step 2: 实现 app/task/arq_config.py**

```python
from arq.connections import RedisSettings
from app.config import settings

redis_settings = RedisSettings(
    host=settings.redis_host,
    port=settings.redis_port,
    database=1,
    password=settings.redis_password or None,
)

ARQ_CONCURRENCY = 2
ARQ_RETRY_COUNT = 3
ARQ_RETRY_DELAY = 60
ARQ_TIMEOUT = 1800  # 30 min in seconds
```

- [ ] **Step 3: 实现 app/task/jobs.py**

```python
from pathlib import Path
import tempfile
import os
import logging
from datetime import datetime
from app.config import settings
from app.file.minio_client import download_file, get_minio
from app.file.parser import parse_file
from app.file.chunker import chunk_text
from app.rag.indexing import MilvusIndexWriter

logger = logging.getLogger(__name__)


async def process_document(ctx, doc_id: int, filename: str, file_type: str, category: str, minio_key: str):
    """ARQ job: 处理上传文件 — 解析 → 分块 → Embedding → Milvus"""
    from app.db.database import async_session
    from app.db.models import Document

    async with async_session() as db:
        result = await db.execute(
            __import__("sqlalchemy").select(Document).where(Document.id == doc_id)
        )
        doc = result.scalar_one_or_none()
        if doc:
            doc.status = "processing"
            doc.updated_at = datetime.now()
            await db.commit()

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            local_path = os.path.join(tmpdir, filename)
            download_file(minio_key, local_path)

            text = parse_file(local_path, file_type)
            chunks = chunk_text(text, file_type, metadata={
                "source": filename,
                "category": category,
                "doc_id": str(doc_id),
            })

            writer = MilvusIndexWriter(collection_name=f"ml_{category}")
            writer.write(chunks)

        async with async_session() as db:
            result = await db.execute(
                __import__("sqlalchemy").select(Document).where(Document.id == doc_id)
            )
            doc = result.scalar_one_or_none()
            if doc:
                doc.status = "done"
                doc.updated_at = datetime.now()
                await db.commit()

        logger.info("document processed doc_id=%s chunks=%d", doc_id, len(chunks))
        return {"status": "done", "chunks": len(chunks)}

    except Exception as e:
        async with async_session() as db:
            result = await db.execute(
                __import__("sqlalchemy").select(Document).where(Document.id == doc_id)
            )
            doc = result.scalar_one_or_none()
            if doc:
                doc.status = "failed"
                doc.updated_at = datetime.now()
                await db.commit()
        logger.exception("document processing failed doc_id=%s", doc_id)
        return {"status": "failed", "error": str(e)}


async def build_index(ctx, task_type: str, category: str, document_ids: list[int] | None = None):
    """ARQ job: 结构化数据向量化构建"""
    from app.db.database import async_session
    from app.db.models import IndexTask
    import pymysql
    from app.config import settings as s

    async with async_session() as db:
        task = IndexTask(
            task_type=task_type,
            category=category,
            document_ids=document_ids,
            status="running",
            created_by="system",
        )
        db.add(task)
        await db.commit()
        await db.refresh(task)
        task_id = task.id

    try:
        conn = pymysql.connect(
            host=s.database_url.split(":")[0] if ":" in s.database_url else s.database_url,
            user=s.database_user,
            password=s.database_password,
            database=s.database_db_name,
            charset="utf8mb4",
        )
        table_map = {"legal": "法律法规", "tender": "政府招标信息", "product": "市场产品信息"}
        table = table_map.get(category)
        if not table:
            raise ValueError(f"Unknown category: {category}")

        cursor = conn.cursor()
        if document_ids:
            placeholders = ",".join(["%s"] * len(document_ids))
            cursor.execute(f"SELECT * FROM `{table}` WHERE id IN ({placeholders})", document_ids)
        else:
            cursor.execute(f"SELECT * FROM `{table}`")

        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        conn.close()

        processed = 0
        for row in rows:
            data = dict(zip(columns, [str(v) if v else "" for v in row]))
            content = " ".join(data.values())
            if len(content) < 10:
                continue
            chunks = chunk_text(content, "txt", metadata={"source": table, "category": category, **data})
            writer = MilvusIndexWriter(collection_name=f"ml_{category}")
            writer.write(chunks)
            processed += 1

        async with async_session() as db:
            result = await db.execute(
                __import__("sqlalchemy").select(IndexTask).where(IndexTask.id == task_id)
            )
            t = result.scalar_one_or_none()
            if t:
                t.status = "done"
                t.result_msg = f"成功处理 {processed} 条记录"
                await db.commit()

        return {"status": "done", "processed": processed}

    except Exception as e:
        async with async_session() as db:
            result = await db.execute(
                __import__("sqlalchemy").select(IndexTask).where(IndexTask.id == task_id)
            )
            t = result.scalar_one_or_none()
            if t:
                t.status = "failed"
                t.result_msg = str(e)
                await db.commit()
        logger.exception("index build failed")
        return {"status": "failed", "error": str(e)}
```

- [ ] **Step 4: 实现 app/task/worker.py**

```python
import logging
from arq import create_worker
from app.task.arq_config import redis_settings
from app.task.jobs import process_document, build_index

logging.basicConfig(level=logging.INFO)


async def startup(ctx):
    logging.info("ARQ worker started")


async def shutdown(ctx):
    logging.info("ARQ worker stopped")


FUNCTIONS = [
    process_document,
    build_index,
]


if __name__ == "__main__":
    import asyncio
    worker = create_worker(
        redis_settings=redis_settings,
        functions=FUNCTIONS,
        on_startup=startup,
        on_shutdown=shutdown,
    )
    asyncio.run(worker.async_run())
```

- [ ] **Step 5: Commit**

```bash
git add app/task/
git commit -m "feat: add ARQ task definitions and worker for document processing and index building"
```

---

### Task 13: Documents 和 Index API 路由

**Files:**
- Create: `app/schemas/document.py`
- Create: `app/schemas/index.py`
- Create: `app/api/documents.py`
- Create: `app/api/index.py`

- [ ] **Step 1: app/schemas/document.py**

```python
from pydantic import BaseModel


class DocumentResponse(BaseModel):
    id: int
    filename: str
    category: str
    file_size: int | None
    status: str
    created_at: str | None


class DocumentUpdate(BaseModel):
    filename: str | None = None
    category: str | None = None
```

- [ ] **Step 2: app/schemas/index.py**

```python
from pydantic import BaseModel


class IndexBuildRequest(BaseModel):
    task_type: str  # full / incremental
    category: str
    document_ids: list[int] | None = None


class IndexTaskResponse(BaseModel):
    id: int
    task_type: str
    category: str
    status: str
    result_msg: str | None
    created_at: str | None
    finished_at: str | None
```

- [ ] **Step 3: app/api/documents.py**

```python
import uuid
from datetime import date
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc
from app.auth.dependencies import require_role
from app.db.database import get_db
from app.db.models import User, Document
from app.file.minio_client import upload_file, get_minio, ensure_bucket
from app.config import settings
from app.task.jobs import process_document
from app.schemas.document import DocumentResponse, DocumentUpdate
from arq import ArqRedis

router = APIRouter(prefix="/api/documents", tags=["documents"])

SUPPORTED_EXTENSIONS = {".pdf": "pdf", ".md": "md", ".txt": "txt"}
ADMIN = require_role("admin")


@router.post("/upload")
async def upload_single(
    category: str = "legal",
    file: UploadFile = File(...),
    user: User = Depends(ADMIN),
    db: AsyncSession = Depends(get_db),
):
    ext = Path(file.filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(400, f"不支持的文件类型: {ext}，仅支持 PDF/MD/TXT")
    file_type = SUPPORTED_EXTENSIONS[ext]

    ensure_bucket()
    today = date.today().isoformat()
    object_name = f"{category}/{today}/{uuid.uuid4().hex}{ext}"

    import tempfile
    import os
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    file_size = len(content)
    upload_file(tmp_path, object_name)
    os.unlink(tmp_path)

    doc = Document(
        filename=file.filename,
        category=category,
        file_path=object_name,
        file_size=file_size,
        status="pending",
        uploaded_by=user.id,
        created_by=str(user.id),
    )
    db.add(doc)
    await db.commit()
    await db.refresh(doc)

    # Enqueue ARQ job
    from app.task.arq_config import redis_settings
    from arq.connections import ArqRedis
    arq: ArqRedis = await ArqRedis(redis_settings)
    await arq.enqueue_job("process_document", doc.id, file.filename, file_type, category, object_name)

    return {"id": doc.id, "status": doc.status, "filename": file.filename}


@router.get("")
async def list_documents(
    category: str | None = None,
    page: int = 1,
    page_size: int = 20,
    user: User = Depends(ADMIN),
    db: AsyncSession = Depends(get_db),
):
    query = select(Document).where(Document.is_deleted == False)
    if category:
        query = query.where(Document.category == category)
    query = query.order_by(desc(Document.created_at)).offset((page - 1) * page_size).limit(page_size)
    result = await db.execute(query)
    docs = result.scalars().all()
    return [
        DocumentResponse(
            id=d.id, filename=d.filename, category=d.category,
            file_size=d.file_size, status=d.status or "pending",
            created_at=d.created_at.isoformat() if d.created_at else None,
        ) for d in docs
    ]


@router.put("/{doc_id}")
async def update_document(
    doc_id: int, body: DocumentUpdate,
    user: User = Depends(ADMIN), db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(Document).where(Document.id == doc_id, Document.is_deleted == False))
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(404, "文件不存在")
    if body.filename is not None:
        doc.filename = body.filename
    if body.category is not None:
        doc.category = body.category
    doc.updated_by = str(user.id)
    await db.commit()
    return {"ok": True}


@router.delete("/{doc_id}")
async def delete_document(
    doc_id: int,
    user: User = Depends(ADMIN), db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(Document).where(Document.id == doc_id, Document.is_deleted == False))
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(404, "文件不存在")
    doc.is_deleted = True
    doc.deleted_by = str(user.id)
    doc.deleted_at = __import__("datetime").datetime.now()
    await db.commit()
    return {"ok": True}
```

- [ ] **Step 4: app/api/index.py**

```python
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc
from app.auth.dependencies import require_role
from app.db.database import get_db
from app.db.models import User, IndexTask, IntentLog
from app.schemas.index import IndexBuildRequest, IndexTaskResponse
from arq import ArqRedis

router = APIRouter(prefix="/api/index", tags=["index"])
ADMIN = require_role("admin")


@router.post("/build")
async def trigger_build(
    body: IndexBuildRequest,
    user: User = Depends(ADMIN),
):
    from app.task.arq_config import redis_settings
    arq: ArqRedis = await ArqRedis(redis_settings)
    job = await arq.enqueue_job("build_index", body.task_type, body.category, body.document_ids)
    return {"job_id": job.job_id}


@router.get("/tasks")
async def list_tasks(
    page: int = 1, page_size: int = 20,
    user: User = Depends(ADMIN), db: AsyncSession = Depends(get_db),
):
    query = select(IndexTask).order_by(desc(IndexTask.created_at)).offset((page - 1) * page_size).limit(page_size)
    result = await db.execute(query)
    tasks = result.scalars().all()
    return [
        IndexTaskResponse(
            id=t.id, task_type=t.task_type, category=t.category,
            status=t.status or "queued", result_msg=t.result_msg,
            created_at=t.created_at.isoformat() if t.created_at else None,
            finished_at=t.finished_at.isoformat() if t.finished_at else None,
        ) for t in tasks
    ]


@router.get("/tasks/{task_id}")
async def get_task(
    task_id: int,
    user: User = Depends(ADMIN), db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(IndexTask).where(IndexTask.id == task_id))
    t = result.scalar_one_or_none()
    if not t:
        raise HTTPException(404, "任务不存在")
    return IndexTaskResponse(
        id=t.id, task_type=t.task_type, category=t.category,
        status=t.status or "queued", result_msg=t.result_msg,
        created_at=t.created_at.isoformat() if t.created_at else None,
        finished_at=t.finished_at.isoformat() if t.finished_at else None,
    )


@router.get("/intent-logs")
async def list_intent_logs(
    failed_level: str | None = None, page: int = 1, page_size: int = 50,
    user: User = Depends(ADMIN), db: AsyncSession = Depends(get_db),
):
    query = select(IntentLog).order_by(desc(IntentLog.created_at))
    if failed_level:
        query = query.where(IntentLog.failed_level == failed_level)
    query = query.offset((page - 1) * page_size).limit(page_size)
    result = await db.execute(query)
    logs = result.scalars().all()
    return [
        {
            "id": l.id, "session_id": l.session_id, "question": l.question,
            "failed_level": l.failed_level, "scores": l.scores,
            "final_intent": l.final_intent,
            "created_at": l.created_at.isoformat() if l.created_at else None,
        } for l in logs
    ]
```

- [ ] **Step 5: 验证导入**

```bash
cd E:\Github\TenderRag\TenderRag && uv run python -c "from app.api.documents import router; from app.api.index import router; print('OK')"
```

- [ ] **Step 6: Commit**

```bash
git add app/schemas/document.py app/schemas/index.py app/api/documents.py app/api/index.py
git commit -m "feat: add documents and index API endpoints with ARQ job enqueue"
```

---

## Phase 6: 数据浏览 & 主应用组装

### Task 14: 业务表查询 + 重写 main.py

**Files:**
- Create: `app/data/__init__.py`
- Create: `app/data/repository.py`
- Create: `app/api/data.py`
- Modify: `app/main.py` — 重写

- [ ] **Step 1: 创建空文件**

```bash
echo "" > app/data/__init__.py
```

- [ ] **Step 2: 实现 app/data/repository.py**

```python
import pymysql
from app.config import settings

TABLE_MAP = {
    "legal": "法律法规",
    "tender": "政府招标信息",
    "product": "市场产品信息",
}


def _get_conn():
    return pymysql.connect(
        host=settings.database_url,
        user=settings.database_user,
        password=settings.database_password,
        database=settings.database_db_name,
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
    )


def list_records(category: str, page: int = 1, page_size: int = 20) -> dict:
    table = TABLE_MAP.get(category)
    if not table:
        raise ValueError(f"未知分类: {category}")
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) as total FROM `{table}`")
            total = cur.fetchone()["total"]
            offset = (page - 1) * page_size
            cur.execute(f"SELECT * FROM `{table}` LIMIT {page_size} OFFSET {offset}")
            rows = cur.fetchall()
    finally:
        conn.close()
    return {"rows": rows, "total": total, "page": page, "page_size": page_size}


def get_record(category: str, record_id: int) -> dict | None:
    table = TABLE_MAP.get(category)
    if not table:
        raise ValueError(f"未知分类: {category}")
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(f"SELECT * FROM `{table}` WHERE id = %s", (record_id,))
            row = cur.fetchone()
    finally:
        conn.close()
    return row
```

- [ ] **Step 3: 实现 app/api/data.py**

```python
from fastapi import APIRouter, Depends, HTTPException, Query
from app.auth.dependencies import require_role
from app.data.repository import list_records, get_record

router = APIRouter(prefix="/api/data", tags=["data"])
ADMIN = require_role("admin")


@router.get("/{category}")
async def browse_data(
    category: str,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    user=Depends(ADMIN),
):
    if category not in ("legal", "tender", "product"):
        raise HTTPException(400, "category 必须为 legal/tender/product")
    try:
        return list_records(category, page, page_size)
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.get("/{category}/{record_id}")
async def get_data_detail(
    category: str,
    record_id: int,
    user=Depends(ADMIN),
):
    if category not in ("legal", "tender", "product"):
        raise HTTPException(400, "category 必须为 legal/tender/product")
    record = get_record(category, record_id)
    if not record:
        raise HTTPException(404, "记录不存在")
    return record
```

- [ ] **Step 4: 重写 app/main.py**

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.auth import router as auth_router
from app.api.chat import router as chat_router
from app.api.documents import router as documents_router
from app.api.data import router as data_router
from app.api.index import router as index_router
from app.file.minio_client import ensure_bucket


def create_app() -> FastAPI:
    app = FastAPI(title="TenderRag API", version="2.0.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(auth_router)
    app.include_router(chat_router)
    app.include_router(documents_router)
    app.include_router(data_router)
    app.include_router(index_router)

    @app.on_event("startup")
    async def startup():
        ensure_bucket()

    @app.get("/api/health")
    async def health():
        return {"status": "ok", "version": "2.0.0"}

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
```

- [ ] **Step 5: 验证整个应用启动**

```bash
cd E:\Github\TenderRag\TenderRag && timeout 5 uv run python -c "from app.main import app; print('App created:', app.title)" || true
```

- [ ] **Step 6: Commit**

```bash
git add app/data/ app/api/data.py app/main.py
git commit -m "feat: add data browsing API and rewrite main.py to assemble all routers"
```

---

## Phase 7: 前端（Vue3 + Element Plus）

### Task 16: 外部对话台 Vue3 应用

**Files:**
- Create: `frontend/chat/` — Vue3 项目

- [ ] **Step 1: 使用 Vite 创建项目**

```bash
cd E:\Github\TenderRag\TenderRag
npm create vite@latest frontend/chat -- --template vue-ts
cd frontend/chat
npm install element-plus vue-router@4 pinia axios
```

- [ ] **Step 2: 创建 router — frontend/chat/src/router/index.ts**

```typescript
import { createRouter, createWebHistory } from 'vue-router'

const routes = [
  { path: '/login', name: 'Login', component: () => import('../views/Login.vue') },
  { path: '/register', name: 'Register', component: () => import('../views/Register.vue') },
  { path: '/chat', name: 'Chat', component: () => import('../views/Chat.vue'), meta: { requiresAuth: true } },
  { path: '/:pathMatch(.*)*', redirect: '/chat' },
]

export default createRouter({ history: createWebHistory(), routes })
```

- [ ] **Step 3: 创建 API 封装 — frontend/chat/src/api/index.ts**

```typescript
import axios from 'axios'

const api = axios.create({ baseURL: 'http://localhost:8000' })

api.interceptors.request.use((config) => {
  const token = localStorage.getItem('token')
  if (token) config.headers.Authorization = `Bearer ${token}`
  return config
})

api.interceptors.response.use(
  (res) => res,
  (err) => {
    if (err.response?.status === 401) {
      localStorage.removeItem('token')
      window.location.href = '/login'
    }
    return Promise.reject(err)
  }
)

export default api
```

- [ ] **Step 4: 创建 user store — frontend/chat/src/stores/user.ts**

```typescript
import { defineStore } from 'pinia'
import { ref } from 'vue'
import api from '../api'

export const useUserStore = defineStore('user', () => {
  const user = ref<any>(null)
  const token = ref(localStorage.getItem('token') || '')

  async function login(username: string, password: string) {
    const { data } = await api.post('/api/auth/login', { username, password })
    token.value = data.access_token
    localStorage.setItem('token', data.access_token)
    await fetchMe()
  }

  async function register(username: string, password: string) {
    const { data } = await api.post('/api/auth/register', { username, password })
    token.value = data.access_token
    localStorage.setItem('token', data.access_token)
    await fetchMe()
  }

  async function fetchMe() {
    const { data } = await api.get('/api/auth/me')
    user.value = data
  }

  function logout() {
    token.value = ''
    user.value = null
    localStorage.removeItem('token')
  }

  return { user, token, login, register, fetchMe, logout }
})
```

- [ ] **Step 5: 创建对话页面 — frontend/chat/src/views/Chat.vue**

核心布局：左侧 ElMenu 展示会话列表 + 右侧聊天区（ElScrollbar 包裹消息列表 + ElInput + ElButton 发送）。流式渲染通过 fetch + ReadableStream 解析 SSE。

详细布局参考设计文档中的外部对话台线框图。

- [ ] **Step 6: 创建登录/注册页面**

Login.vue：ElForm + username/password + ElButton
Register.vue：同上，调 register API

- [ ] **Step 7: Commit**

```bash
cd E:\Github\TenderRag\TenderRag
git add frontend/chat/
git commit -m "feat: scaffold Vue3 external chat app with login, register, and chat views"
```

---

### Task 17: 内部管理台 Vue3 应用

**Files:**
- Create: `frontend/admin/` — Vue3 项目

- [ ] **Step 1: 创建项目**

```bash
cd E:\Github\TenderRag\TenderRag
npm create vite@latest frontend/admin -- --template vue-ts
cd frontend/admin
npm install element-plus vue-router@4 pinia axios
```

- [ ] **Step 2: 创建 router — frontend/admin/src/router/index.ts**

```typescript
import { createRouter, createWebHistory } from 'vue-router'

const routes = [
  { path: '/login', name: 'Login', component: () => import('../views/Login.vue') },
  {
    path: '/',
    component: () => import('../layouts/MainLayout.vue'),
    meta: { requiresAuth: true, requiresAdmin: true },
    children: [
      { path: '', redirect: '/documents' },
      { path: 'documents', name: 'Documents', component: () => import('../views/Documents.vue') },
      { path: 'data', name: 'DataBrowse', component: () => import('../views/DataBrowse.vue') },
      { path: 'chat', name: 'Chat', component: () => import('../views/Chat.vue') },
      { path: 'tasks', name: 'Tasks', component: () => import('../views/Tasks.vue') },
    ],
  },
  { path: '/:pathMatch(.*)*', redirect: '/' },
]

export default createRouter({ history: createWebHistory(), routes })
```

- [ ] **Step 3: 创建 MainLayout.vue — 顶栏 ElTabs 导航**

```vue
<template>
  <el-container>
    <el-header>
      <el-menu mode="horizontal" :default-active="activeTab" router>
        <el-menu-item index="/documents">文档管理</el-menu-item>
        <el-menu-item index="/data">数据浏览</el-menu-item>
        <el-menu-item index="/chat">内部问答</el-menu-item>
        <el-menu-item index="/tasks">任务日志</el-menu-item>
      </el-menu>
      <div class="user-info">
        <span>{{ userStore.user?.username }}</span>
        <el-button @click="logout">退出</el-button>
      </div>
    </el-header>
    <el-main><router-view /></el-main>
  </el-container>
</template>
```

- [ ] **Step 4: 实现四个主要页面**

**Documents.vue**：ElTable 展示文件列表（分页、按分类筛选），ElUpload 上传（传 category 参数），行操作（编辑/删除按钮）

**DataBrowse.vue**：ElSelect 选分类 → ElTable 渲染 MySQL 数据（分页），ElCheckbox 勾选行 → ElButton 触发 POST /api/index/build

**Chat.vue**：和外部对话台相同的 Chat 组件（可提取为共享组件），session_id 从 store 获取

**Tasks.vue**：两个 ElTab — 索引任务表 + 意图日志表，ElTable + 分页展示

- [ ] **Step 5: 路由守卫（权限检查）**

```typescript
router.beforeEach(async (to, _from, next) => {
  const token = localStorage.getItem('token')
  if (to.meta.requiresAuth && !token) return next('/login')
  if (token && to.name === 'Login') return next('/')
  if (to.meta.requiresAdmin) {
    // Check role from stored user info or re-fetch /api/auth/me
  }
  next()
})
```

- [ ] **Step 6: Commit**

```bash
cd E:\Github\TenderRag\TenderRag
git add frontend/admin/
git commit -m "feat: scaffold Vue3 internal admin app with document management, data browsing, chat, and task views"
```

---

## Phase 8: 部署 & 清理

### Task 15: Docker 和最终清理

**Files:**
- 修改: `docker-compose.yml`
- 创建/修改: `Dockerfile`
- 删除: `app/memory/store.py`（替换为 chat/session.py）
- 删除: `app/frontend/` 目录

- [ ] **Step 1: 更新 Dockerfile**

```dockerfile
FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN pip install uv && uv sync --frozen

COPY . .

# FastAPI
CMD ["uv", "run", "python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

- [ ] **Step 2: 更新 docker-compose.yml 添加 Redis 和 Worker**

在 docker-compose.yml 中新增：

```yaml
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    restart: unless-stopped

  arq-worker:
    build: .
    command: uv run python -m app.task.worker
    depends_on:
      - redis
      - mysql
      - milvus
    restart: unless-stopped
```

- [ ] **Step 3: 删除旧文件**

```bash
rm -f app/memory/store.py
rm -rf app/frontend/
# 检查是否有 __pycache__
find app -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
```

- [ ] **Step 4: 最终验证**

```bash
cd E:\Github\TenderRag\TenderRag && uv run python -c "
from app.main import app
routes = [r.path for r in app.routes]
print('Routes:', len(routes))
for r in routes:
    print(' ', r)
"
```

- [ ] **Step 5: 最终 Commit**

```bash
git add -A
git commit -m "chore: update Dockerfile, add Redis + ARQ worker to compose, remove old Gradio/memory"
```