import logging
from arq import run_worker
from app.task.arq_config import (
    redis_settings,
    ARQ_CONCURRENCY,
    ARQ_RETRY_COUNT,
    ARQ_TIMEOUT,
)
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
    worker = run_worker(
        redis_settings=redis_settings,
        functions=FUNCTIONS,
        on_startup=startup,
        on_shutdown=shutdown,
        max_jobs=ARQ_CONCURRENCY,
        job_timeout=ARQ_TIMEOUT,
        max_tries=ARQ_RETRY_COUNT,
    )
    asyncio.run(worker)