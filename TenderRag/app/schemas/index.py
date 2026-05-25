from pydantic import BaseModel


class IndexBuildRequest(BaseModel):
    task_type: str  # full / incremental
    category: str
    document_ids: list[int] | None = None


class IndexTaskResponse(BaseModel):
    id: int
    task_type: str
    category: str
    status: str
    result_msg: str | None
    created_at: str | None