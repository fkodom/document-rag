from __future__ import annotations

import os
from typing import Any, List, Sequence, Tuple, cast

from qdrant_client import QdrantClient
from typing_extensions import Self

from document_rag.vector_db.base import BaseVectorDB, SearchResult, TextMetadata

# TODO: Move to configurable Settings class
COLLECTION_NAME = "documents"


class QdrantVectorDB(BaseVectorDB):
    """Implementation of a Qdrant vector DB, which is consistent with the
    BaseVectorDB interface.
    """

    def __init__(self, client: QdrantClient):
        self.client = client

    @classmethod
    def create(cls, cache_dir: str, exist_ok: bool = False) -> Self:
        os.makedirs(cache_dir, exist_ok=exist_ok)
        return cls(client=QdrantClient(path=cache_dir))

    def add_documents(self, documents: Sequence[Tuple[str, TextMetadata]]) -> None:
        """Add one or more documents to the DB, along with associated metadata."""
        _documents = [doc for doc, _ in documents]
        metadata = [metadata for _, metadata in documents]
        self.client.add(
            collection_name=COLLECTION_NAME,
            documents=_documents,
            metadata=cast(list[dict[str, Any]], metadata),  # For mypy
        )

    def search(self, query: str, limit: int = 10) -> List[SearchResult]:
        """Query the DB, and return up to 'limit' most similar results.

        Args:
            query: The query text.
            limit: The maximum number of results to return.

        Returns:
            A list of search results, sorted by similarity in decreasing order.
        Raises:
            ValueError: If the DB is empty.
        """
        collection_info = self.client.get_collection(collection_name=COLLECTION_NAME)
        if collection_info.segments_count == 0:
            raise ValueError("The DB is empty.")

        results = self.client.query(
            collection_name=COLLECTION_NAME, query_text=query, limit=limit
        )
        return [
            SearchResult(
                text=result.document,
                similarity=result.score,
                metadata=cast(TextMetadata, result.metadata),
            )
            for result in results
        ]
