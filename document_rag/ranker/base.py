from abc import abstractmethod
from typing import List, Sequence


class BaseRanker:
    @abstractmethod
    def predict(self, query: str, documents: Sequence[str]) -> List[float]:
        """Predict the relevance of sequence of documents, based on the given query."""
