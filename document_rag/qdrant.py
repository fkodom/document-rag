import logging
import os
import shutil
from typing import Tuple, TypedDict

from pypdf import PdfReader
from qdrant_client import QdrantClient

# TODO: Move these to configurable Settings class
COLLECTION_NAME = "documents"
INDEX_PATH = os.path.join("data", "qdrant")
CHUNK_SIZE = 256
CHUNK_OVERLAP = 64


class TextInfo(TypedDict):
    """Metadata to attach to each chunk of text in the Qdrant index."""

    path: str
    page_range: Tuple[int, int]


def _format_text(text: str) -> str:
    """Standard text formatting, applied to every chunk before adding to the index."""
    return (
        text.replace("\n\r", " ")
        .replace("\n", " ")
        .replace("\t", " ")
        .replace("- ", "-")
        .strip("-")
        .strip(" ")
    )


def get_qdrant_index(index_path: str, overwrite: bool = False) -> QdrantClient:
    """Builds a Qdrant index with local data cache, and returns the Qdrant client.

    Args:
        index_path (str): Path to the index.
        overwrite (bool, optional): Whether to overwrite the existing index. Defaults to True.

    Returns:
        QdrantClient: Qdrant client.
    """
    os.makedirs(index_path, exist_ok=True)

    if os.path.exists(index_path) and not overwrite:
        logging.info(f"Loading existing Qdrant index at {index_path=}")
    else:
        if overwrite:
            logging.info(f"Removing existing Qdrant index at {index_path=}")
            shutil.rmtree(index_path)
        else:
            logging.info(f"Creating new Qdrant index at {index_path=}")
        os.makedirs(index_path, exist_ok=True)

    return QdrantClient(path=index_path)


def add_pdf_document_to_index(
    path: str,
    client: QdrantClient,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> None:
    _, ext = os.path.splitext(path)
    if ext.lower() != ".pdf":
        raise ValueError(f"File extension '{ext}' not supported. Must be PDF.")

    reader = PdfReader(path)
    num_pages = len(reader.pages)
    current_page = 0
    start_page = 0
    words = []

    documents: list[str] = []
    metadata: list[TextInfo] = []

    while current_page < num_pages:
        if len(words) < chunk_size:
            page = reader.pages[current_page]
            words += _format_text(page.extract_text()).split(" ")
            current_page += 1
        else:
            text = " ".join(words[:chunk_size])
            words = words[chunk_size - chunk_overlap :]
            documents.append(text)
            metadata.append(TextInfo(path=path, page_range=(start_page, current_page)))

            start_page = current_page

    if len(words) > 0:
        text = " ".join(words)
        documents.append(text)
        metadata.append(TextInfo(path=path, page_range=(start_page, current_page)))

    client.add(
        collection_name=COLLECTION_NAME,
        documents=documents,
        metadata=metadata,
    )


if __name__ == "__main__":
    client = get_qdrant_index(INDEX_PATH, overwrite=True)
    add_pdf_document_to_index(
        path="assets/alice-in-wonderland-short.pdf",
        client=client,
        chunk_size=64,
        chunk_overlap=16,
    )
    results = client.query(
        collection_name=COLLECTION_NAME,
        # query_text="What does the rabbit have in its pocket?",
        query_text="Who is in Wonderland?",
        limit=5,
    )
    for result in results:
        print("-" * 10)
        print(f"score: {result.score}, path: '{result.metadata['path']}")
        print(result.document)
