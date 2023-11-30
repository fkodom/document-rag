from typing import Tuple, TypedDict


class TextMetadata(TypedDict):
    """Metadata to attach to each chunk of text in the vector index."""

    path: str
    page_range: Tuple[int, int]


class SearchResult(TypedDict):
    """Description of a single text chunk, which is returned by a vector DB search."""

    text: str
    similarity: float
    metadata: TextMetadata
