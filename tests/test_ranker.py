from typing import Optional, Type

import pytest

from document_rag.ranker import load_ranker
from document_rag.ranker.huggingface import HuggingFaceRanker


@pytest.mark.parametrize(
    "type, model, error",
    [
        ("huggingface", "cross-encoder/ms-marco-TinyBERT-L-2-v2", None),
        ("huggingface", "cross-encoder/ms-marco-MiniLM-L-6-v2", None),
        ("unsupported-type", "cross-encoder/ms-marco-MiniLM-L-6-v2", ValueError),
        ("huggingface", "model-does-not-exist", OSError),  # error from HF backend
    ],
)
def test_load_ranker(type: str, model: str, error: Optional[Type[Exception]]):
    if error is not None:
        with pytest.raises(error):
            _ = load_ranker(type=type, model=model)
    else:
        _ = load_ranker(type=type, model=model)


@pytest.fixture(scope="session")
def ranker() -> HuggingFaceRanker:
    return HuggingFaceRanker(model="cross-encoder/ms-marco-MiniLM-L-6-v2")


def test_predict(ranker: HuggingFaceRanker):
    scores = ranker.predict(
        query="How many people live in Berlin?",
        documents=[
            "Berlin has a population of 3,520,031 registered inhabitants in an area of 891.82 square kilometers.",
            "New York City is famous for the Metropolitan Museum of Art.",
        ],
    )
    assert len(scores) == 2
    assert scores[0] > scores[1]
