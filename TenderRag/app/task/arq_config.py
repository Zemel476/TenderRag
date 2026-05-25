from arq.connections import RedisSettings, create_pool
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

_pool = None


async def get_arq_pool():
    global _pool
    if _pool is None:
        _pool = await create_pool(redis_settings)
    return _pool


async def enqueue_job(job_name: str, *args):
    pool = await get_arq_pool()
    return await pool.enqueue_job(job_name, *args)