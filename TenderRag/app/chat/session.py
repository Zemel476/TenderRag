import json
from datetime import datetime, timezone
from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession as SqlAlchemyAsyncSession
from app.db.models import Session, Message
from app.config import settings
import redis.asyncio as aioredis
import redis as sync_redis

redis_client = aioredis.from_url(
    f"redis://{settings.redis_host}:{settings.redis_port}/{settings.redis_db}",
    password=settings.redis_password or None,
    decode_responses=True,
)

# Sync Redis client for graph nodes (which run in threads)
sync_redis_client = sync_redis.Redis(
    host=settings.redis_host,
    port=settings.redis_port,
    db=settings.redis_db,
    password=settings.redis_password or None,
    decode_responses=True,
)


class SessionManager:
    def __init__(self, db: SqlAlchemyAsyncSession):
        self.db = db

    async def create_session(self, user_id: int, title: str = "新对话") -> Session:
        session = Session(
            user_id=user_id,
            title=title,
            created_by=str(user_id),
        )
        self.db.add(session)
        await self.db.commit()
        await self.db.refresh(session)
        return session

    async def list_sessions(self, user_id: int) -> list[dict]:
        result = await self.db.execute(
            select(Session)
            .where(Session.user_id == user_id, Session.is_deleted == False)
            .order_by(desc(Session.updated_at))
        )
        sessions = result.scalars().all()
        return [
            {
                "id": s.id,
                "title": s.title or "新对话",
                "created_at": s.created_at.isoformat(),
            }
            for s in sessions
        ]

    async def get_messages(self, session_id: int, limit: int = 20) -> list[dict]:
        cache_key = f"session:{session_id}:messages"
        cached = await redis_client.lrange(cache_key, 0, -1)
        if cached:
            return [json.loads(m) for m in cached]

        result = await self.db.execute(
            select(Message)
            .where(Message.session_id == session_id, Message.is_deleted == False)
            .order_by(Message.created_at)
            .limit(limit)
        )
        messages = result.scalars().all()
        data = [
            {
                "role": m.role,
                "content": m.content,
                "intents": m.intents,
                "created_at": m.created_at.isoformat(),
            }
            for m in messages
        ]

        for item in data:
            await redis_client.rpush(cache_key, json.dumps(item, ensure_ascii=False))
        await redis_client.expire(cache_key, 3600)
        return data

    async def add_message(
        self,
        session_id: int,
        role: str,
        content: str,
        intents: list[str] | None = None,
    ):
        msg = Message(
            session_id=session_id,
            role=role,
            content=content,
            intents=intents,
            created_by="system",
        )
        self.db.add(msg)
        await self.db.commit()

        cache_key = f"session:{session_id}:messages"
        item = {
            "role": role,
            "content": content,
            "intents": intents,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        await redis_client.rpush(cache_key, json.dumps(item, ensure_ascii=False))
        await redis_client.expire(cache_key, 3600)

        result = await self.db.execute(select(Session).where(Session.id == session_id))
        session = result.scalar_one_or_none()
        if session:
            session.updated_at = datetime.now(timezone.utc)
            await self.db.commit()

    async def get_session_owner(self, session_id: int) -> int | None:
        """Return user_id of the session owner, or None if not found."""
        result = await self.db.execute(
            select(Session).where(Session.id == session_id, Session.is_deleted == False)
        )
        session = result.scalar_one_or_none()
        return session.user_id if session else None

    async def soft_delete_session(self, session_id: int, deleted_by: str):
        result = await self.db.execute(
            select(Session).where(Session.id == session_id, Session.is_deleted == False)
        )
        session = result.scalar_one_or_none()
        if session:
            session.is_deleted = True
            session.deleted_at = datetime.now(timezone.utc)
            session.deleted_by = deleted_by
            await self.db.commit()

        cache_key = f"session:{session_id}:messages"
        await redis_client.delete(cache_key)