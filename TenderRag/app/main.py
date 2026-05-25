import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.auth import router as auth_router
from app.api.chat import router as chat_router
from app.api.documents import router as documents_router
from app.api.data import router as data_router
from app.api.index import router as index_router
from app.config import settings
from app.file.minio_client import ensure_bucket

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_bucket()
    if settings.jwt_secret_key == "change-me-in-production":
        logger.warning(
            "SECURITY: JWT secret key is still the default value. "
            "Set JWT_SECRET_KEY in .env before production deployment."
        )
    yield


def create_app() -> FastAPI:
    app = FastAPI(title="TenderRag API", version="2.0.0", lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(auth_router)
    app.include_router(chat_router)
    app.include_router(documents_router)
    app.include_router(data_router)
    app.include_router(index_router)

    @app.get("/api/health")
    async def health():
        return {"status": "ok", "version": "2.0.0"}

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)