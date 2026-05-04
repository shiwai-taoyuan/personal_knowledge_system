from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    openai_api_key: str = ""
    openai_base_url: Optional[str] = None
    chat_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    checkpoints_dir: str = "checkpoints"
    bge_model_path: str = "checkpoints/bge-small-zh-v1d5"
    qwen_model_path: str = "checkpoints/Qwen3-0d6B"
    embedding_provider: str = "local_bge"
    chat_provider: str = "local_qwen"
    vector_index_dir: str = "data/index"
    docs_dir: str = "data/docs"
    chunk_size: int = 500
    chunk_overlap: int = 100
    min_chunk_chars: int = 120
    ingest_rebuild_default: bool = True
    default_top_k: int = 4
    retrieval_candidate_k: int = 12
    retrieval_score_threshold: float = 0.2
    reranker_type: str = "similarity"
    reranker_model: str = "BAAI/bge-reranker-base"
    max_per_source: int = 2
    max_context_chars: int = 4000
    temperature: float = 0.2
    qwen_disable_thinking: bool = True

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
