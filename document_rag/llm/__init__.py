from enum import Enum
from typing import Union

from document_rag.llm.base import BaseLLM


class LLMType(str, Enum):
    OPENAI = "openai"
    HUGGINGFACE = "huggingface"


def load_llm(type: Union[LLMType, str], model: str) -> BaseLLM:
    if isinstance(type, str):
        type = LLMType(type)

    # fmt: off
    if type == LLMType.OPENAI:
        from document_rag.llm.openai import OpenAILLM
        return OpenAILLM(model=model)
    elif type == LLMType.HUGGINGFACE:
        from document_rag.llm.huggingface import HuggingFaceLLM
        return HuggingFaceLLM(model=model)
    else:
        raise ValueError(f"Unknown LLM type: {type}")
    # fmt: on
