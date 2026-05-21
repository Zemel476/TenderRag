# TenderRag — 招投标智能问答系统

基于 LangGraph + LlamaIndex + Milvus 的 RAG 问答系统，覆盖法律法规、招标公告、产品信息等领域，支持混合检索与流式对话。

## 功能特性

- **多领域覆盖**：法律条文、招标公告、产品信息、招投标流程
- **混合检索**：Embedding 向量 + BM25 关键词 + RRF 融合，提升召回质量
- **流式输出**：Gradio 前端和 FastAPI SSE 均支持逐字流式响应
- **多轮对话**：基于会话历史的上下文感知问答
- **意图分类**：LLM 自动识别用户意图并路由到对应领域 Agent
- **Docker 部署**：提供 Dockerfile 和 docker-compose 一键启动

## 架构

```
Gradio 前端 (:7860) ──→ FastAPI 后端 (:8000)
                                ↓
                       LangGraph StateGraph
                       ├── memory_retrieve    获取历史对话
                       ├── classify_intent     LLM 意图分类
                       ├── route_intent        条件路由
                       │   ├── legal_agent     法律检索
                       │   ├── tender_agent    招标检索
                       │   ├── product_agent   产品检索
                       │   ├── bidding_agent   招投标流程
                       │   └── other_node      无关问题
                       ├── synthesize          整合回答 (流式)
                       └── store_memory        存储对话记忆
                                ↓
                       Milvus VectorStore
                       ├── ml_legal
                       ├── ml_tender
                       └── ml_product
```

## 快速开始

### 1. 环境准备

- Python >= 3.12
- [uv](https://docs.astral.sh/uv/) 包管理器
- MySQL（原始数据源）
- Milvus（向量存储）

### 2. 安装依赖

```bash
uv sync
```

### 3. 配置环境变量

```bash
cp .explame.env .env
```

编辑 `.env`，填入以下必要配置：

```env
# LLM 模型 (OpenAI 兼容接口)
MODEL_NAME=qwen-plus
MODEL_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
MODEL_API_KEY=your-api-key

# Rerank 模型
RERANK_MODEL_NAME=gte-rerank-v2
RERANK_MODEL_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
RERANK_MODEL_API_KEY=your-api-key

# Embedding 模型
EMBEDDING_MODEL_NAME=text-embedding-v4
EMBEDDING_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
EMBEDDING_API_KEY=your-api-key

# MySQL
DATABASE_URL=127.0.0.1
DATABASE_USER=root
DATABASE_PASSWORD=123456
DATABASE_DB_NAME=tenderrag

# Milvus
MILVUS_VECTOR_URI=http://127.0.0.1:19530
MILVUS_VECTOR_LEGAL_NAME=ml_legal
MILVUS_VECTOR_TENDER_NAME=ml_tender
MILVUS_VECTOR_PRODUCT_NAME=ml_product
```

### 4. 构建向量索引

```bash
uv run python scripts/vector_index_legal.py
uv run python scripts/vector_index_tender.py
uv run python scripts/vector_index_product.py
```

### 5. 启动服务

```bash
# FastAPI 后端 (端口 8000)
uv run python -m app.main

# Gradio 前端 (端口 7860，另一个终端)
uv run python -m app.frontend.gradio_app
```

### 6. 访问

| 地址 | 说明 |
|------|------|
| http://localhost:7860 | Gradio 聊天界面 |
| http://localhost:8000/docs | API 文档 (Swagger) |
| http://localhost:8000/health | 健康检查 |

## Docker 部署

```bash
# 完整环境 (应用 + MySQL + Milvus)
docker compose up -d

# 仅构建应用镜像
docker build -t tenderrag .
docker run -p 7860:7860 --env-file .env tenderrag
```

## 项目结构

```
app/
├── main.py                     # FastAPI 入口
├── config.py                   # pydantic-settings 配置管理
├── api/
│   └── endpoints.py            # SSE 流式 + 非流式聊天接口
├── agents/
│   ├── graph.py                # LangGraph StateGraph 定义
│   ├── nodes.py                # 图节点实现
│   └── prompts.py              # Prompt 模板
├── rag/
│   ├── hybrid.py               # Embedding + BM25 + RRF 混合检索
│   ├── fusion.py               # RRF 融合排序
│   ├── indexing.py             # 索引构建
│   ├── milvus.py               # MilvusVectorStore 封装
│   ├── legal.py                # 法律领域检索入口
│   ├── tender.py               # 招标领域检索入口
│   ├── product.py              # 产品领域检索入口
│   └── bidding.py              # 招投标流程检索
├── memory/
│   └── store.py                # 会话记忆存储 (内存)
├── models/
│   └── llm.py                  # LLM + Embedding 模型封装
├── schemas/
│   └── chat.py                 # API 请求/响应模型
├── frontend/
│   └── gradio_app.py           # Gradio 聊天界面
└── utils/
    ├── logger.py               # 日志工具
    ├── db_util.py              # 数据库工具
    ├── pdf_load_util.py        # PDF 加载工具
    └── legal_document_util.py  # 法律文档处理

scripts/
├── vector_index_legal.py       # 法律向量索引构建
├── vector_index_tender.py      # 招标向量索引构建
├── vector_index_product.py     # 产品向量索引构建
├── download_model.py           # 模型下载
├── generate_legal_qa.py        # 法律问答数据生成
├── export_legal_nodes_for_qa.py # 法律节点导出
└── check_embedding_dim.py      # 嵌入维度检测

data/
└── input/                      # 原始数据文件
```

## API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/chat` | SSE 流式聊天 |
| POST | `/api/chat/non-stream` | 非流式聊天 (JSON 响应) |
| GET | `/health` | 健康检查 |

### SSE 事件类型

| type | 说明 |
|------|------|
| `thinking` | 思考中 |
| `intent` | 意图分类结果 |
| `message` | 回答内容 (逐字流式) |
| `done` | 完成 |

## 技术栈

| 技术 | 用途 |
|------|------|
| FastAPI | 后端框架 |
| LangGraph | RAG 流程编排 (状态图) |
| LlamaIndex | 文档加载、分块、向量索引 |
| Milvus | 向量数据库 (持久化) |
| PyMuPDF | PDF 文本提取 |
| Gradio 6.x | Web 聊天界面 |
| PyMySQL | MySQL 数据库连接 |
| uv | Python 包管理器 |