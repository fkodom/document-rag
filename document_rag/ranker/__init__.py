from enum import Enum
from typing import Union

from document_rag.ranker.base import BaseRanker


class RankerType(str, Enum):
    HUGGINGFACE = "huggingface"


def load_ranker(type: Union[RankerType, str], model: str) -> BaseRanker:
    if isinstance(type, str):
        type = RankerType(type)

    # fmt: off
    if type == RankerType.HUGGINGFACE:
        from document_rag.ranker.huggingface import HuggingFaceRanker
        return HuggingFaceRanker(model=model)
    else:
        raise ValueError(f"Unknown ranker type: {type}")
    # fmt: on
