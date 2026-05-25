from pydantic import BaseModel


class DocumentResponse(BaseModel):
    id: int
    filename: str
    category: str
    file_size: int | None
    status: str
    created_at: str | None


class DocumentUpdate(BaseModel):
    filename: str | None = None
    category: str | None = None