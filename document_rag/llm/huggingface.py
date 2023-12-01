from typing import Optional, TypedDict, cast

from transformers import pipeline

from document_rag.llm.base import BaseLLM


class Response(TypedDict):
    """Typed description of the response from the Transformers model"""

    generated_text: str


class HuggingFaceLLM(BaseLLM):
    def __init__(
        self, model: str, do_sample: bool = False, token: Optional[str] = None
    ):
        self.model = model
        self.do_sample = do_sample
        self.pipeline = pipeline("text-generation", model=self.model, token=token)

    def generate(self, prompt: str) -> str:
        """Generate text from a prompt using the OpenAI API."""
        response = cast(Response, self.pipeline(prompt, do_sample=False)[0])
        return response["generated_text"]
