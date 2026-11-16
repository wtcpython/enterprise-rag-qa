FROM ghcr.io/astral-sh/uv:python3.14-bookworm-slim

WORKDIR /app

# 复制依赖配置并利用 uv 缓存高速安装依赖
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# 复制项目源码与静态数据
COPY app ./app
COPY eval ./eval
COPY data ./data
COPY main.py run.sh ./

EXPOSE 8000

# 启动 FastAPI 服务
CMD ["uv", "run", "--no-sync", "uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]

