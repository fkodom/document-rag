import shutil

import pytest

from document_rag.rag import RAG
from document_rag.settings import Settings


# Monkey-patch the 'rag.generate' method to return dummy data.
# So we don't accidentally incur a bunch of charges from automated testing.
@pytest.fixture(autouse=True)
def mock_generate(monkeypatch, rag: RAG):
    def mock_llm_generate(*args, **kwargs):
        return "Alice"

    monkeypatch.setattr(rag.llm, "generate", mock_llm_generate)


@pytest.fixture(scope="session")
def rag() -> RAG:
    shutil.rmtree(Settings().DOCUMENT_RAG_VECTOR_DB_CACHE_DIR, ignore_errors=True)
    return RAG.from_settings()


def test_add_pdf_documents(rag: RAG):
    with pytest.raises(ValueError):
        rag.generate("What is Wonderland?")

    rag.add_pdf_documents(paths=["assets/alice-in-wonderland-short.pdf"])

    with pytest.raises(FileNotFoundError):
        rag.add_pdf_documents(paths=["assets/does-not-exist.pdf"])

    with pytest.raises(ValueError):
        rag.add_pdf_documents(paths=["assets/alice-in-wonderland.txt"])


def test_vector_db_already_exists():
    with pytest.raises(FileExistsError):
        _ = RAG.from_settings()


def test_generate(rag: RAG):
    _ = rag.generate(prompt="What is the name of Alice's cat?")
