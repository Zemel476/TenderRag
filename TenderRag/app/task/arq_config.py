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