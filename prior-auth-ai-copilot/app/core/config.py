from pydantic_settings import BaseSettings
from pydantic import Field
from functools import lru_cache


class Settings(BaseSettings):
    # App
    app_name: str = Field(default="prior-auth-ai-copilot")
    env: str = Field(default="development")
    log_level: str = Field(default="INFO")

    # OpenAI
    openai_api_key: str = Field(default="")

    # Models
    llm_model: str = Field(default="gpt-4o-mini")
    embedding_model: str = Field(default="text-embedding-3-small")
    default_max_tokens: int = Field(default=4096)
    default_temperature: float = Field(default=0.0)

    # Vector store
    vector_store_path: str = Field(default="data/processed/vector_store")
    chroma_collection_name: str = Field(default="prior_auth_policies")

    # Database
    sqlite_db_path: str = Field(default="data/audit.db")

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


@lru_cache()
def get_settings() -> Settings:
    """
    Returns a cached Settings instance.
    Use this everywhere instead of instantiating Settings directly.
    """
    return Settings()
