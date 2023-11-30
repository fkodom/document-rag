from enum import Enum
from typing import Union

from document_rag.vector_db.base import (  # noqa: F401
    BaseVectorDB,
    SearchResult,
    TextMetadata,
)


class VectorDBType(str, Enum):
    QDRANT = "qdrant"


def create_vector_db(
    type: Union[VectorDBType, str], cache_dir: str, exist_ok: bool = False
) -> BaseVectorDB:
    if isinstance(type, str):
        type = VectorDBType(type)

    # fmt: off
    if type == VectorDBType.QDRANT:
        from document_rag.vector_db.qdrant import QdrantVectorDB
        return QdrantVectorDB.create(cache_dir=cache_dir, exist_ok=exist_ok)
    else:
        raise ValueError(f"Unknown vector DB type: {type}")
    # fmt: on
