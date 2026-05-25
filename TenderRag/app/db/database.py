from urllib.parse import quote_plus

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker

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


async def get_db() -> AsyncSession:
    async with async_session() as session:
        yield session