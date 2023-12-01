from __future__ import annotations

import os
from typing import List, Sequence, Tuple, cast

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
        return QdrantVectorDB(client=QdrantClient(path=cache_dir))

    def add_documents(self, documents: Sequence[Tuple[str, TextMetadata]]) -> None:
        """Add one or more documents to the DB, along with associated metadata."""
        _documents = [doc for doc, _ in documents]
        metadata = [metadata for _, metadata in documents]
        self.client.add(
            collection_name=COLLECTION_NAME,
            documents=_documents,
            metadata=metadata,
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
                metadata=cast(SearchResult, result.metadata),
            )
            for result in results
        ]


if __name__ == "__main__":
    import shutil

    from document_rag.vector_db.base import INDEX_PATH

    shutil.rmtree(INDEX_PATH, ignore_errors=True)
    db: QdrantVectorDB = QdrantVectorDB.create(INDEX_PATH, exist_ok=True)
    db.add_pdf_documents(["assets/alice-in-wonderland-short.pdf"])
    results = db.search("Who is in Wonderland?", limit=5)
    for result in results:
        print("-" * 10)
        print(
            f"similarity: {result['similarity']}, path: '{result['metadata']['path']}"
        )
        print(result["text"])
