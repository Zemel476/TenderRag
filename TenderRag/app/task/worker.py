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