import os
from typing import Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Settings for the document_rag package."""

    OPENAI_API_KEY: Optional[str] = None
    HUGGINGFACE_TOKEN: Optional[str] = None

    DOCUMENT_RAG_VERSION: str = "0.0.0"

    # TODO: Add descriptions for what these settings do.
    DOCUMENT_RAG_CHUNK_SIZE: int = 128
    DOCUMENT_RAG_CHUNK_OVERLAP: int = 32
    DOCUMENT_RAG_RETRIEVER_CHUNKS: int = 100
    DOCUMENT_RAG_RANKER_CHUNKS: int = 5

    DOCUMENT_RAG_LLM_TYPE: str = "openai"
    DOCUMENT_RAG_LLM_MODEL: str = "gpt-3.5-turbo-1106"
    DOCUMENT_RAG_RANKER_TYPE: str = "huggingface"
    DOCUMENT_RAG_RANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    DOCUMENT_RAG_VECTOR_DB_TYPE: str = "qdrant"
    DOCUMENT_RAG_VECTOR_DB_CACHE_DIR: str = os.path.join("data", "vector_db")

    class Config:
        env_file = ".env"
