from typing import List, Optional, Sequence, TypedDict

import numpy as np
from typing_extensions import Self

from document_rag.llm import BaseLLM, load_llm
from document_rag.ranker import BaseRanker, load_ranker
from document_rag.settings import Settings
from document_rag.vector_db import BaseVectorDB, SearchResult, create_vector_db

SETTINGS = Settings()
DOCUMENT_TEMPLATE = """
(similarity={similarity})
{text}
"""
PROMPT_TEMPLATE = """Answer a question based on a collection of documents.

QUESTION: {question}

DOCUMENTS:

{documents}

END OF DOCUMENTS

Base your answer ONLY on the documents above.  Answer as concisely as possible, while
still being complete.  If you cannot answer, respond with the word UNKNOWN."""


class RAGResult(TypedDict):
    text: str
    prompt: str
    search_results: Sequence[SearchResult]


class RAG:
    """Simple implementation of retrieval-augmented generation (RAG), which is
    inter-operable with several types of LLMs, text ranking models, and vector DBs.
    """

    def __init__(
        self,
        llm: BaseLLM,
        ranker: BaseRanker,
        vector_db: BaseVectorDB,
        # TODO: Add description for these parameters
        retriever_chunks: int = SETTINGS.DOCUMENT_RAG_RETRIEVER_CHUNKS,
        ranker_chunks: int = SETTINGS.DOCUMENT_RAG_RANKER_CHUNKS,
    ):
        self.llm = llm
        self.ranker = ranker
        self.vector_db = vector_db
        self.retriever_chunks = retriever_chunks
        self.ranker_chunks = ranker_chunks

    @classmethod
    def from_settings(
        cls,
        settings: Optional[Settings] = None,
        vector_db_exists_ok: bool = False,
    ) -> Self:
        """Instantiates a RAG object from the given settings.  All Settings fields
        are configurable through ENV variables, and are loaded automatically. Users
        may choose to modify them at the ENV level, or pass in a custom Settings
        object to this method.

        Args:
            settings: The Settings object to use.  If None, a default Settings object
                will be created.
            vector_db_exists_ok: Whether to allow the vector DB to already exist on
                disk.  If False, an error will be raised if the DB already exists.
                This ensures that we don't accidentally write document embeddings
                to an existing DB, which may contain unrelated data.
        """
        if not settings:
            settings = Settings()

        llm = load_llm(
            type=settings.DOCUMENT_RAG_LLM_TYPE,
            model=settings.DOCUMENT_RAG_LLM_MODEL,
        )
        ranker = load_ranker(
            type=settings.DOCUMENT_RAG_RANKER_TYPE,
            model=settings.DOCUMENT_RAG_RANKER_MODEL,
        )
        vector_db = create_vector_db(
            type=settings.DOCUMENT_RAG_VECTOR_DB_TYPE,
            cache_dir=settings.DOCUMENT_RAG_VECTOR_DB_CACHE_DIR,
            exist_ok=vector_db_exists_ok,
        )

        return cls(llm=llm, ranker=ranker, vector_db=vector_db)

    def add_pdf_documents(self, paths: Sequence[str], verbose: bool = False) -> None:
        """Add one or more PDF documents to the DB, keeping track of text metadata.

        Args:
            paths: The local paths to the PDF documents to add.
            verbose: Whether to display a progress bar during the PDF extraction step.
        """
        self.vector_db.add_pdf_documents(paths, verbose=verbose)

    # TODO: Move number of documents to a configurable setting
    def generate(self, prompt: str) -> RAGResult:
        """Run retrieval-augmented generation on a prompt, using the given documents.

        Args:
            prompt: The question or prompt from a user, which will be used as the
                input to the LLM.

        Returns:
            The generated response from the LLM.
        """
        retriever_results = self.vector_db.search(prompt, limit=self.retriever_chunks)
        ranker_scores = self.ranker.predict(
            documents=[result["text"] for result in retriever_results],
            query=prompt,
        )
        sorted_indices = np.argsort(ranker_scores).tolist()
        topk_indices = sorted_indices[-self.ranker_chunks :]
        ranker_results: List[SearchResult] = [
            {**retriever_results[i], "similarity": ranker_scores[i]}  # type: ignore
            for i in topk_indices
        ]

        document_strings = [
            DOCUMENT_TEMPLATE.format(
                similarity=result["similarity"], text=result["text"]
            )
            for result in ranker_results
        ]
        documents = "\n".join(document_strings)
        llm_prompt = PROMPT_TEMPLATE.format(documents=documents, question=prompt)
        llm_response = self.llm.generate(llm_prompt)

        return RAGResult(
            text=llm_response,
            prompt=llm_prompt,
            search_results=ranker_results,
        )
