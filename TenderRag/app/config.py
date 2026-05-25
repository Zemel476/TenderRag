from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    debug: bool = True

    # 大模型
    model_name: str = ""
    model_base_url: str = ""
    model_api_key: str = ""

    # rerank模型
    rerank_model_name: str = ""
    rerank_base_url: str = ""
    rerank_api_key: str = ""

    # 向量模型
    embedding_model_name: str = ""
    embedding_base_url: str = ""
    embedding_api_key: str = ""

    # Database
    database_url: str = "127.0.0.1"
    database_user: str = "root"
    database_password: str = "123456"
    database_db_name: str = ""

    # milvus_vector
    milvus_vector_uri: str = ""
    # milvus_db_name
    milvus_vector_legal_name: str = "ml_legal"
    milvus_vector_tender_name: str = "ml_tender"
    milvus_vector_product_name: str = "ml_product"

    # Redis
    redis_host: str = "127.0.0.1"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: str = ""

    # Local file storage (replaces MinIO)
    storage_dir: str = "data/uploads"

    # JWT
    jwt_secret_key: str = "change-me-in-production"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 1440

    # Intent Pipeline
    jieba_threshold: float = 0.6
    bert_threshold: float = 0.7
    llm_threshold: float = 0.5
    bert_model_path: str = "models/bert_intent_classifier.pth"
    bert_model_type: str = "bert-base-chinese"
    num_intent_labels: int = 5
    intent_labels: list[str] = ["legal", "tender", "bidding", "product", "other"]

    # ARQ
    arq_redis_dsn: str = "redis://127.0.0.1:6379/1"

    # Database (for SQLAlchemy)
    database_async_url: str = ""


settings = Settings()
