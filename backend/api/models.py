"""Request models shared by the HTTP API routes."""

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    conversation_id: str | None = Field(default=None, min_length=1, max_length=64)
    use_web: bool = False
    generate_image: bool = False


class SessionRequest(BaseModel):
    id_token: str = Field(min_length=20, max_length=10000)


class EditExchangeRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


class IndexRequest(BaseModel):
    filename: str = Field(
        min_length=1,
        description="PDF, TXT, or Markdown filename inside the project's data directory.",
    )


class YouTubeIndexRequest(BaseModel):
    url: str = Field(
        min_length=1,
        max_length=2000,
        description="YouTube video or playlist URL.",
    )


class FolderIndexRequest(BaseModel):
    folder_path: str = Field(
        min_length=1,
        max_length=4096,
        description="Server-local folder under one of INDEX_FOLDER_ROOTS.",
    )
    recursive: bool = Field(
        default=True,
        description="Include supported documents in nested folders.",
    )
