from urllib.parse import quote_plus

from sqlalchemy import create_engine as create_sync_engine
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import Session as SyncSession, sessionmaker as sync_sessionmaker

from app.config import settings

DATABASE_ASYNC_URL = settings.database_async_url or (
    f"mysql+asyncmy://{quote_plus(settings.database_user)}:{quote_plus(settings.database_password)}"
    f"@{settings.database_url}/{settings.database_db_name}"
)

engine = create_async_engine(
    DATABASE_ASYNC_URL,
    echo=settings.debug,
    pool_size=10,
    max_overflow=20,
)

async_session = async_sessionmaker(engine, expire_on_commit=False)

# Sync engine for graph nodes (which run in threads)
DATABASE_SYNC_URL = (
    f"mysql+pymysql://{quote_plus(settings.database_user)}:{quote_plus(settings.database_password)}"
    f"@{settings.database_url}/{settings.database_db_name}?charset=utf8mb4"
)

_sync_engine = create_sync_engine(
    DATABASE_SYNC_URL,
    echo=settings.debug,
    pool_size=5,
    max_overflow=10,
)

SyncSessionLocal = sync_sessionmaker(_sync_engine, expire_on_commit=False)


async def get_db() -> AsyncSession:
    async with async_session() as session:
        yield session