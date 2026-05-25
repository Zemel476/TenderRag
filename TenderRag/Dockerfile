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

EXPOSE 8000

ENV PYTHONUNBUFFERED=1

# 默认启动 FastAPI
CMD ["uv", "run", "python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]