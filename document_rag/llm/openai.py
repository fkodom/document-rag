from enum import Enum
from typing import List, TypedDict, Union

from openai import OpenAI

from document_rag.llm.base import BaseLLM
from document_rag.settings import Settings

SETTINGS = Settings()


class ModelType(str, Enum):
    GPT_3_5_TURBO = "gpt-3.5-turbo"
    GPT_3_5_TURBO_1106 = "gpt-3.5-turbo-1106"
    GPT_4_TURBO_1106 = "gpt-4-1106-preview"


class Message(TypedDict):
    role: str
    content: str


class Choice(TypedDict):
    finish_reason: str
    index: int
    message: Message


class Response(TypedDict):
    """Typed description of the response from the OpenAI API"""

    # TODO: Add other response fields.  See:
    # https://platform.openai.com/docs/guides/text-generation/chat-completions-response-format
    choices: List[Choice]


class OpenAILLM(BaseLLM):
    def __init__(self, model: Union[ModelType, str]):
        if isinstance(model, str):
            model = ModelType(model)
        self.model = model
        self.client = OpenAI(api_key=SETTINGS.OPENAI_API_KEY)

    def generate(self, prompt: str) -> str:
        """Generate text from a prompt using the OpenAI API."""
        openai_response = self.client.chat.completions.create(
            model=self.model.value,
            messages=[{"role": "user", "content": prompt}],
        )
        return openai_response.choices[0].message.content


if __name__ == "__main__":
    response = OpenAILLM(
        model=ModelType.GPT_3_5_TURBO,
    ).generate("Respond with just the word 'STOP'.")
    print(response)
