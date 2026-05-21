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


settings = Settings()
