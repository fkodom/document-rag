from typing import Optional, Sequence, cast

import numpy as np
from sentence_transformers import CrossEncoder

from document_rag.ranker.base import BaseRanker


class HuggingFaceRanker(BaseRanker):
    def __init__(self, model: str, device: Optional[str] = None):
        self.model = CrossEncoder(model, device=device)

    def predict(self, query: str, documents: Sequence[str]) -> list[float]:
        """Predict the relevance of a query and a list of documents."""
        scores = cast(
            np.ndarray,
            self.model.predict([(query, document) for document in documents]),
        )
        return scores.tolist()
