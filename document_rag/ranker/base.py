from abc import abstractmethod
from typing import Sequence


class BaseRanker:
    @abstractmethod
    def predict(self, query: str, documents: Sequence[str]) -> list[float]:
        """Predict the relevance of sequence of documents, based on the given query."""
