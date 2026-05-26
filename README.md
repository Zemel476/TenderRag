# TenderRag — 招投标智能问答系统

基于 LangGraph + LlamaIndex + Milvus 的 RAG 问答系统，覆盖法律法规、招标公告、产品信息、招投标流程等领域，支持混合检索与流式对话。

## 功能特性

- **多领域覆盖**：法律条文、招标公告、产品信息、招投标流程
- **混合检索**：Embedding 向量 + BM25 关键词 + RRF 融合，提升召回质量
- **多级意图识别**：Jieba 关键词 → BERT 模型 → LLM 兜底，三级管道逐步降级
- **双端架构**：Vue3 内部管理台 + Vue3 外部对话台，角色权限分离
- **流式输出**：FastAPI SSE 逐字流式响应
- **多轮对话**：MySQL + Redis 持久化会话历史
- **文件管理**：PDF/MD/TXT 上传 → 解析分块 → 向量化入 Milvus
- **异步任务**：ARQ (Redis) 后台处理文档索引和结构化数据向量化
- **Docker 部署**：提供 Dockerfile 和 docker-compose 一键启动

## 架构

```
Vue3 内部管理台 (:5173) ──┐
                           ├── FastAPI (:8000) ── MySQL / Redis / Milvus / 本地存储
Vue3 外部对话台 (:5174) ──┘
                                │
                       LangGraph StateGraph
                       ├── memory_retrieve    MySQL 历史消息
                       ├── classify_intent     Jieba → BERT → LLM
                       ├── route_intent        条件路由
                       │   ├── legal_agent     法律检索
                       │   ├── tender_agent    招标检索
                       │   ├── bidding_agent   投标检索
                       │   └── product_agent   产品检索
                       ├── merge_contexts      扇入合并
                       ├── synthesize          整合回答 (流式)
                       └── store_memory        MySQL + Redis 持久化
```

## 快速开始

### 1. 环境准备

- Python >= 3.12
- [uv](https://docs.astral.sh/uv/) 包管理器
- MySQL 8.x（业务数据 + 应用数据）
- Milvus 2.6.x（向量存储）
- Redis 7.x（缓存 + 任务队列）
- 本地文件系统（文件存储于 `app/data/uploads/`）

### 2. 安装依赖

```bash
cd TenderRag
uv sync
```

### 3. 配置环境变量

```bash
cp .explame.env .env
```

编辑 `.env`，填入必要配置：

```env
# LLM 模型 (OpenAI 兼容接口)
MODEL_NAME=qwen-plus
MODEL_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
MODEL_API_KEY=your-api-key

# MySQL
DATABASE_URL=127.0.0.1
DATABASE_USER=root
DATABASE_PASSWORD=123456
DATABASE_DB_NAME=tenderrag

# Milvus
MILVUS_VECTOR_URI=http://127.0.0.1:19530

# Redis
REDIS_HOST=127.0.0.1
REDIS_PORT=6379

# JWT (生产环境务必修改)
JWT_SECRET_KEY=your-secret-key
```

### 4. 初始化数据库

```bash
uv run python -m app.db.init_db
```

> 向量索引构建已集成到管理台 UI：登录管理台 → 数据浏览 → 选择分类 → 点击"构建索引"。也可通过 API `POST /api/index/build` 触发后台 ARQ 任务。

### 5. 启动服务

```bash
# FastAPI 后端 (端口 8000)
uv run python -m app.main

# ARQ Worker (另一个终端)
uv run python -m app.task.worker
```

### 6. 启动前端

```bash
cd frontend/chat
npm install && npm run dev      # 外部对话台 → :5173

cd frontend/admin
npm install && npm run dev      # 内部管理台 → :5174
```

### 7. 访问

| 地址 | 说明 |
|------|------|
| http://localhost:5173 | 外部对话台 (注册/登录/对话) |
| http://localhost:5174 | 内部管理台 (文档管理/数据浏览/问答/日志) |
| http://localhost:8000/docs | API 文档 (Swagger) |
| http://localhost:8000/api/health | 健康检查 |

## Docker 部署

```bash
docker compose up -d
```

启动服务：FastAPI + ARQ Worker + Redis + MySQL + Milvus（前端需单独构建或挂载 nginx）。

## 项目结构

```
TenderRag/                       # Python 后端
├── app/
│   ├── main.py                  # FastAPI 入口
│   ├── config.py                # pydantic-settings 配置管理
│   ├── api/
│   │   ├── auth.py              # 注册/登录/用户信息
│   │   ├── chat.py              # SSE 流式 + 非流式聊天 + 会话管理
│   │   ├── documents.py         # 文件上传/列表/编辑/删除
│   │   ├── data.py              # 业务表数据浏览
│   │   └── index.py             # 索引任务 + 意图日志
│   ├── auth/
│   │   ├── service.py           # JWT 签发/校验 + 注册登录
│   │   └── dependencies.py      # get_current_user / require_role
│   ├── db/
│   │   ├── database.py          # SQLAlchemy async engine
│   │   ├── models.py            # ORM 模型 (6 张新表)
│   │   └── init_db.py           # 建表脚本
│   ├── intent/
│   │   ├── base.py              # BaseIntentClassifier 抽象类
│   │   ├── jieba_classifier.py  # Level 1: Jieba 关键词打分
│   │   ├── bert_classifier.py   # Level 2: BERT 模型分类
│   │   ├── llm_classifier.py    # Level 3: LLM 兜底
│   │   └── pipeline.py          # 三级管道编排
│   ├── agents/
│   │   ├── graph.py             # LangGraph StateGraph 定义
│   │   ├── nodes.py             # 图节点实现
│   │   └── prompts.py           # Prompt 模板
│   ├── rag/                     # 检索核心
│   │   ├── hybrid.py, fusion.py, milvus.py, indexing.py
│   │   └── legal.py, tender.py, bidding.py, product.py
│   ├── chat/
│   │   └── session.py           # Session + Message CRUD (MySQL + Redis)
│   ├── file/
│   │   ├── minio_client.py      # 文件存储客户端
│   │   ├── parser.py            # PDF/MD/TXT 解析
│   │   └── chunker.py           # 分块策略
│   ├── task/
│   │   ├── arq_config.py        # ARQ Redis 配置
│   │   ├── jobs.py              # 后台任务定义 (文档处理 + 索引构建)
│   │   └── worker.py            # Worker 启动入口
│   ├── data/
│   │   ├── repository.py        # 业务表查询
│   │   └── uploads/             # 上传文件存储
│   ├── models/llm.py            # LLM + Embedding 模型封装
│   ├── schemas/                 # Pydantic 请求/响应模型
│   │   ├── auth.py, chat.py, document.py, index.py
│   └── utils/                   # 工具函数 (PDF 加载、数据库工具等)
│
frontend/                        # 前端
├── chat/                        # Vue3 外部对话台
│   └── src/views/
│       ├── Login.vue, Register.vue, Chat.vue
└── admin/                       # Vue3 内部管理台
    └── src/views/
        ├── Login.vue, Documents.vue, DataBrowse.vue, Chat.vue, Tasks.vue

evaluation/                      # RAG 评测
├── embedding/                   # Embedding 模型评测
└── model/                       # 模型评测
```

## API 接口

### Auth
| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/auth/register` | 注册 (公开) |
| POST | `/api/auth/login` | 登录返回 JWT (公开) |
| GET | `/api/auth/me` | 当前用户信息 (需登录) |

### Chat
| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/chat` | SSE 流式聊天 |
| POST | `/api/chat/non-stream` | 非流式聊天 |
| GET | `/api/sessions` | 对话列表 |
| POST | `/api/sessions` | 创建对话 |
| GET | `/api/sessions/{id}/messages` | 对话消息 |
| DELETE | `/api/sessions/{id}` | 删除对话 |

### Documents (admin only)
| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/documents/upload` | 上传文件 |
| GET | `/api/documents` | 文件列表 |
| PUT | `/api/documents/{id}` | 编辑元数据 |
| DELETE | `/api/documents/{id}` | 删除文件 |

### Data (admin only)
| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/data/{category}` | 分页浏览业务表 |
| GET | `/api/data/{category}/{id}` | 单条详情 |

### Index & Logs (admin only)
| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/index/build` | 触发索引任务 |
| GET | `/api/index/tasks` | 任务列表 |
| GET | `/api/index/tasks/{id}` | 任务详情 |
| GET | `/api/intent-logs` | 意图识别日志 |

## 技术栈

| 技术 | 用途 |
|------|------|
| FastAPI | 后端框架 |
| LangGraph | RAG 流程编排 (状态图) |
| LlamaIndex | 文档加载、分块、向量索引 |
| Milvus | 向量数据库 |
| SQLAlchemy + asyncmy | ORM + 异步 MySQL |
| Redis | 会话缓存 + ARQ 任务队列 |
| ARQ | 异步任务队列 |
| 本地存储 | 文件上传与存储 (data/uploads/) |
| PyTorch + Transformers | BERT 意图分类 |
| Jieba | 中文分词 + 关键词提取 |
| Vue3 + Element Plus | 前端框架 |
| PyMuPDF | PDF 文本提取 |
| Docker | 容器化部署 |
| uv | Python 包管理器 |