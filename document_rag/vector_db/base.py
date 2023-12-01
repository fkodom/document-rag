from __future__ import annotations

import os
from abc import abstractmethod
from typing import Any, Callable, List, Sequence, Tuple, TypeVar

from pypdf import PdfReader
from tqdm import tqdm
from typing_extensions import Self

from document_rag.settings import Settings
from document_rag.types import SearchResult, TextMetadata

T = TypeVar("T")

SETTINGS = Settings()
CHUNK_SIZE = SETTINGS.DOCUMENT_RAG_CHUNK_SIZE
CHUNK_OVERLAP = SETTINGS.DOCUMENT_RAG_CHUNK_OVERLAP


class BaseVectorDB:
    """Base class for vector DBs.  Abstracts away the details of the underlying DB,
    so that the rest of the code can be agnostic to the DB implementation.

    NOTE: For simplicity, assume that all DBs have a local cache, and that the
    cache is contained to a single folder within the project directory.  (Not true
    in general, but we could generalize this later if needed.)
    """

    @classmethod
    @abstractmethod
    def create(cls, cache_dir: str, exist_ok: bool = False) -> Self:
        pass

    @abstractmethod
    def add_documents(self, documents: Sequence[Tuple[str, TextMetadata]]) -> None:
        """Add one or more documents to the DB, along with associated metadata."""
        """Add a document to the DB, along with associated metadata."""

    @abstractmethod
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

    def add_pdf_documents(self, paths: Sequence[str], verbose: bool = False) -> None:
        """Add one or more PDF documents to the DB, keeping track of text metadata.

        TODO:
        - Allow passing custom encoder/decoder functions at this level.  For
          simplicity, we just use the default ones for now.
        - Parallelize the PDF extraction step.  I expect this to be a bottleneck,
          both for speed and memory footprint.
        """
        extracted: List[Tuple[str, TextMetadata]] = []
        for path in tqdm(paths, disable=(not verbose), desc="Extracting PDFs"):
            extracted += read_pdf_document(path)

        self.add_documents(extracted)


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


def read_pdf_document(
    path: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    preprocessor: Callable[[str], str] = _format_text,
    encoder: Callable[[str], list] = lambda x: x.split(" "),
    decoder: Callable[[Sequence], str] = lambda x: " ".join(x),
) -> List[Tuple[str, TextMetadata]]:
    """Extracts text from a PDF document, keeping track of which page numbers each
    chunk of text came from.  The page range is contained in the metadata for each
    text chunk.

    NOTE: By default, the chunk size is measured in words -- not characters or tokens.
    This choice is agnostic to the language models that are used downstream and
    reasonably fast, since we can just split on whitespace.  Other tokenizers (e.g.
    HuggingFace tokenizers) can be used by passing custom encoder/decoder functions.
    """
    result: List[Tuple[str, TextMetadata]] = []

    _, ext = os.path.splitext(path)
    if ext.lower() != ".pdf":
        raise ValueError(f"File extension '{ext}' not supported. Must be PDF.")
    elif not os.path.exists(path):
        raise FileNotFoundError(f"File '{path}' does not exist.")

    reader = PdfReader(path)
    num_pages = len(reader.pages)
    current_page = 0
    start_page = 0
    tokens: List[Any] = []

    while current_page < num_pages:
        # If we don't have enough words to fill a chunk, add the next page of words.
        # Otherwise, add the chunk to the output and continue to the next one.
        if len(tokens) < chunk_size:
            text = preprocessor(reader.pages[current_page].extract_text())
            tokens += encoder(preprocessor(text))
            current_page += 1
        else:
            text = decoder(tokens[:chunk_size])
            tokens = tokens[chunk_size - chunk_overlap :]
            result.append(
                (text, TextMetadata(path=path, page_range=(start_page, current_page)))
            )
            start_page = current_page

    # Add the last chunk of words to the output.
    if len(tokens) > 0:
        text = decoder(tokens)
        result.append(
            (text, TextMetadata(path=path, page_range=(start_page, current_page)))
        )

    return result
