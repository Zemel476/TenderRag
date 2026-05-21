# syntax=docker/dockerfile:1
FROM python:3.12-slim

WORKDIR /app

# 安装系统依赖 (pymupdf 需要)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libmupdf-dev \
    && rm -rf /var/lib/apt/lists/*

# 安装 uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# 先复制依赖文件，利用 Docker 层缓存
COPY pyproject.toml uv.lock ./

# 安装 Python 依赖
RUN uv sync --frozen --no-dev --no-install-project

# 复制项目代码
COPY app/ ./app/
COPY scripts/ ./scripts/
COPY data/ ./data/

# 复制环境变量模板
COPY .explame.env ./.explame.env

EXPOSE 7860 8000

ENV PYTHONUNBUFFERED=1

# 默认运行 Gradio 前端，可通过 CMD 覆盖为 FastAPI
# docker run ... tenderrag                           → Gradio UI (端口 7860)
# docker run ... tenderrag uvicorn app.main:app ...  → FastAPI (端口 8000)
CMD ["uv", "run", "python", "-m", "app.frontend.gradio_app"]