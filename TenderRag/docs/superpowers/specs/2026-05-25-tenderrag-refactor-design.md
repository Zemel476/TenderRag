# TenderRag 重构设计文档

## 概述

将现有 Gradio 单页面 RAG 系统重构为双端架构：内部管理台 + 外部对话台。核心检索栈保持不变（LangGraph + LlamaIndex + Milvus），模块化重写外围系统。

## 技术选型

| 决策项 | 选择 |
|--------|------|
| 后端框架 | FastAPI |
| 图编排 | LangGraph |
| 检索框架 | LlamaIndex |
| 向量库 | Milvus 2.6.x |
| 任务队列 | ARQ (Redis) |
| 前端 | Vue3 + Element Plus |
| 认证 | JWT，三角色（admin / internal / external） |
| 对话存储 | Redis 热数据 + MySQL 持久化 |
| 文件存储 | MinIO（复用已有） |
| 意图分类 BERT | 用户自训 PyTorch 模型 |

## 架构

```
Vue3 内部管理台 (:5173) ──┐
                           ├── FastAPI (:8000) ── MySQL / Redis / Milvus / MinIO
Vue3 外部对话台 (:5174) ──┘
```

### 后端模块

| 模块 | 职责 |
|------|------|
| `auth/` | JWT 签发/校验，用户注册登录，角色权限 |
| `intent/` | 三级意图分类管道（Jieba → BERT → LLM） |
| `rag/` | 现有检索核心（hybrid, fusion, milvus, indexing），不动 |
| `chat/` | LangGraph 图编排，session 管理，多轮对话 |
| `file/` | 文件上传/下载，MinIO 操作 |
| `task/` | ARQ 任务定义与调度（索引更新、文件处理） |
| `data/` | MySQL 业务表数据查询接口 |

## 权限模型

| 角色 | 内部管理台 | 外部对话台 |
|------|-----------|-----------|
| admin | 全部功能 | 可用 |
| internal | 内部问答 | — |
| external | — | 可用（注册登录、对话） |

admin 拥有内部管理台全部权限（文档管理、数据浏览、内部问答、任务日志）。internal 角色仅限内部问答模块，无法访问文档管理、数据浏览、任务日志等管理功能。外部对话台仅 external 角色可登录使用。

## 多级意图识别

```
用户问题
  │
  ▼
Level 1: Jieba 关键词打分（毫秒级）
  ├─ 分数 ≥ JIEBA_THRESHOLD → 直接分类 ✓
  └─ 分数 < JIEBA_THRESHOLD → 记录日志 → 降级
                                │
                                ▼
Level 2: BERT 自训模型分类（~10ms）
  ├─ 置信度 ≥ BERT_THRESHOLD → 命中 ✓
  └─ 置信度 < BERT_THRESHOLD → 记录日志 → 降级
                                │
                                ▼
Level 3: LLM 兜底判断（~1s）
  ├─ 置信度 ≥ LLM_THRESHOLD → 兜底成功 ✓
  └─ 置信度 < LLM_THRESHOLD → 记录日志 → 默认 "other"
```

- BERT 模型：`.pth` / `.bin` 格式，启动时加载，预留 `BaseIntentClassifier` 抽象类
  - 输入：`str`（用户原始问题）
  - 输出：`dict[str, float]`（四类别置信度，如 `{"legal": 0.85, "tender": 0.03, "product": 0.10, "other": 0.02}`）
  - 通过配置指定模型路径和类别映射，方便后续替换升级
- 每级未命中均记录到 `intent_logs` 表，含 question、failed_level、scores、final_intent
- 三级阈值均可通过配置文件调整

## LangGraph 图结构

```
START → memory_retrieve → classify_intent(三级管道) → route_intent
                                                          │
                          ┌───────────────────────────────┤
                          │           │           │       │
                    legal_agent  tender_agent  product_agent  (other→skip)
                          │           │           │       │
                          └───────────┴───────────┴───────┘
                                                          │
                                              merge_contexts → synthesize → store_memory → END
```

变更点：
- `classify_intent`：LLM 单一判断 → 三级管道 + 日志
- `memory_retrieve`：内存 dict → MySQL messages 表
- `store_memory`：内存 dict → Redis 热写入 + MySQL 持久化
- domain agents、merge、synthesize 不变

## 数据库

### 新增表（统一带审计字段）

所有新表含：`created_by`, `created_at`, `updated_by`, `updated_at`, `deleted_by`, `deleted_at`, `is_deleted`

| 表 | 说明 | 关键字段 |
|----|------|---------|
| `users` | 用户 | id, username, password_hash, role(admin/internal/external) |
| `sessions` | 对话会话 | id, user_id(FK), title |
| `messages` | 对话消息 | id, session_id(FK), role(user/assistant), content, intents(JSON) |
| `intent_logs` | 意图识别日志 | id, session_id, question, failed_level(L1/L2/L3), scores(JSON), final_intent |
| `documents` | 文件管理 | id, filename, category, file_path, file_size, status, uploaded_by(FK) |
| `index_tasks` | 索引任务 | id, task_type(full/incremental), category, document_ids(JSON), status, result_msg |

### 原有业务表

`法律法规`、`政府招标信息`、`市场产品信息` 三张表暂不改动。

## API

### Auth
- `POST /api/auth/register` — 注册（公开）
- `POST /api/auth/login` — 登录返回 JWT（公开）
- `GET /api/auth/me` — 当前用户信息（需登录）

### Chat（登录即可）
- `POST /api/chat` — 流式问答 SSE
- `POST /api/chat/non-stream` — 非流式问答
- `GET /api/sessions` — 对话列表
- `GET /api/sessions/{id}/messages` — 对话消息
- `DELETE /api/sessions/{id}` — 删除对话

### Documents（admin only）
- `POST /api/documents/upload` — 上传单文件
- `POST /api/documents/upload/batch` — 批量上传
- `GET /api/documents` — 文件列表
- `PUT /api/documents/{id}` — 修改元数据
- `DELETE /api/documents/{id}` — 删除文件
- `GET /api/documents/{id}/download` — 下载

### Data（admin only）
- `GET /api/data/{category}` — 分页浏览业务表（legal/tender/product）
- `GET /api/data/{category}/{id}` — 单条详情

### Index & Logs（admin only）
- `POST /api/index/build` — 触发索引任务
- `GET /api/index/tasks` — 任务列表
- `GET /api/index/tasks/{id}` — 任务详情
- `GET /api/intent-logs` — 意图日志

## 文件处理

支持三类文件，不同解析和切分策略：

| 类型 | 解析 | 切分 |
|------|------|------|
| PDF | PyMuPDF + 字体分析 → TOC 检测 | 复用现有章/节/条层级切分 |
| Markdown | 读文本，保留 `#` `##` `###` 标题层级 | 按标题层级切分 |
| TXT | 读文本 | 双换行分段 → 长段落滑动窗口切分 |

### 上传流程

1. 用户选择类别 → 上传文件 → `POST /api/documents/upload`
2. 文件存 MinIO（key: `{category}/{date}/{uuid}.ext`），MySQL 记录 status=pending
3. ARQ enqueue → Worker 下载文件 → 按类型解析/切分 → embedding → Milvus
4. 更新 status=done/failed，BM25 缓存自动失效

### 结构化数据向量构建

1. 浏览 MySQL 业务表 → 勾选行 → 选择全量/增量
2. ARQ Worker 读取 MySQL → 复用现有 index 脚本的 chunk/embed 逻辑 → Milvus
3. 前端轮询任务状态

### Worker 配置

| 参数 | 默认值 |
|------|--------|
| max_concurrent | 2 |
| retry_count | 3 |
| retry_delay | 60s |
| timeout | 30min |
| chunk_size | 512 |
| chunk_overlap | 64 |

## 前端

### 内部管理台 (:5173)
- 登录（无注册入口，账号由 admin 创建）
- 文档管理：类别筛选、上传/批量上传、编辑/删除
- 数据浏览：三张业务表分页展示、勾选向量化
- 内部问答：和外部同一套 RAG
- 任务日志：索引任务列表 + 意图识别日志

权限：admin 全部可见，internal 角色仅可用内部问答。

### 外部对话台 (:5174)
- 注册/登录
- 左侧对话列表（新建/切换/删除）
- 右侧聊天区（流式 markdown 渲染）

## 部署

docker-compose 扩展现有配置：
- 新增 Redis 服务
- 新增 ARQ worker 服务（独立进程，与 FastAPI 共享同一代码库，通过 Redis 队列通信）
- 两个 Vue3 前端独立 nginx 容器部署
- 后端服务增加 worker 进程