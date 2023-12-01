from typing import Optional, cast

import numpy as np
from sentence_transformers import CrossEncoder

from document_rag.ranker.base import BaseRanker


class HuggingFaceRanker(BaseRanker):
    def __init__(self, model: str, device: Optional[str] = None):
        self.model = CrossEncoder(model, device=device)

    def predict(self, query: str, documents: list[str]) -> list[float]:
        """Predict the relevance of a query and a list of documents."""
        scores = cast(
            np.ndarray,
            self.model.predict([(query, document) for document in documents]),
        )
        return scores.tolist()


if __name__ == "__main__":
    ranker = HuggingFaceRanker(model="cross-encoder/ms-marco-MiniLM-L-6-v2")
    # query: 'How many people live in Berlin?'
    # documents: ['Berlin has a population of 3,520,031 registered inhabitants in an area of 891.82 square kilometers.', 'New York City is famous for the Metropolitan Museum of Art.']
    scores = ranker.predict(
        query="How many people live in Berlin?",
        documents=[
            "Berlin has a population of 3,520,031 registered inhabitants in an area of 891.82 square kilometers.",
            "New York City is famous for the Metropolitan Museum of Art.",
        ],
    )
    print(scores)
