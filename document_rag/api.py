from fastapi import FastAPI
from pydantic import BaseModel

from document_rag.rag import RAG
from document_rag.settings import Settings

SETTINGS = Settings()
RAG_MODEL = RAG.from_settings()
app = FastAPI(
    title="Document RAG",
    description="Answer questions based on a collection of documents",
    version=SETTINGS.DOCUMENT_RAG_VERSION,
)


@app.get("/")
async def root():
    return {"message": "OK"}


# Add new PDF documents from file uploads
# Save them to a temporary directory, so users can download them later
#
# @app.post("/v1/documents/add")
# @app.get("/v1/documents/get")


class RAGResponse(BaseModel):
    text: str


@app.post("/v1/generate")
async def generate(prompt: str) -> RAGResponse:
    return RAG_MODEL.generate(prompt=prompt)
