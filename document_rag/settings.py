import os
from typing import Optional

from pydantic_settings import BaseSettings

from document_rag import VERSION


class Settings(BaseSettings):
    """Settings for the document_rag package.

    All settings are configurable through ENV variables, and are loaded automatically
    when the Settings class is instantiated.  Users may also create a .env file in
    the root directory of the project to set these variables.  For example:

    ```
    # .env
    OPENAI_API_KEY="..."
    DOCUMENT_RAG_CHUNK_SIZE=96
    ```

    Users may also modify the Settings object directly, before passing it to the
    relevant classes/functions in this project.  For example:

    ```
    settings = Settings()
    settings.DOCUMENT_RAG_CHUNK_SIZE = 96
    settings.DOCUMENT_RAG_CHUNK_OVERLAP = 32
    ```
    """

    class Config:
        env_file = ".env"

    DOCUMENT_RAG_VERSION: str = VERSION

    # Tokens / API keys for third-party services
    OPENAI_API_KEY: Optional[str] = None
    HUGGINGFACE_TOKEN: Optional[str] = None

    # General RAG settings -- applicable to all LLM / ranker / vector DB choices
    #
    # The size of each chunk, measured roughly in words (separated by spaces).
    # The last chunk from each document will typically be smaller.
    DOCUMENT_RAG_CHUNK_SIZE: int = 128
    # The number of words to overlap between each chunk. This helps to ensure that
    # continuous thoughts are fully captured in at least one chunk. (They may be split
    # across multiple chunks, but at least one chunk will contain the full thought.)
    DOCUMENT_RAG_CHUNK_OVERLAP: int = 64
    # The number of initial chunks to retrieve from the vector DB.  We want this to
    # be a relatively large number, so that we have high recall.  These will be
    # filtered down to a smaller number of high-precision chunks by the ranker.
    DOCUMENT_RAG_RETRIEVER_CHUNKS: int = 100
    # The number of chunks to retain after filtering by the ranker.  We want this to
    # be a relatively small number, so that it easily fits into the LLM's context
    # window.  The ranker should be able to filter down to a small number of chunks
    # while maintaining high precision.
    DOCUMENT_RAG_RANKER_CHUNKS: int = 5

    # LLM settings
    #
    # The type of LLM to use.  Currently, only 'openai' and 'huggingface' are supported.
    DOCUMENT_RAG_LLM_TYPE: str = "openai"
    # The name of the LLM model to use.  This is dependent on the LLM type.
    # For more details, see the 'document_rag/llm' directory.
    DOCUMENT_RAG_LLM_MODEL: str = "gpt-3.5-turbo-1106"

    # Ranker settings
    #
    # The type of ranker to use.  Currently, only 'huggingface' is supported.
    DOCUMENT_RAG_RANKER_TYPE: str = "huggingface"
    # The name of the ranker model to use.  This is dependent on the ranker type.
    # For more details, see the 'document_rag/ranker' directory.
    DOCUMENT_RAG_RANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Vector DB settings
    #
    # The type of vector DB to use.  Currently, only 'qdrant' is supported.
    DOCUMENT_RAG_VECTOR_DB_TYPE: str = "qdrant"
    # The directory to use for the vector DB cache.  This directory will be created
    # if it does not already exist. The vector DB will store its data in this
    # directory, and can be (optionally) reloaded in the future.
    DOCUMENT_RAG_VECTOR_DB_CACHE_DIR: str = os.path.join("data", "vector_db")
