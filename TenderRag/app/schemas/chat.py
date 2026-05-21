from pydantic import BaseModel, field_validator


class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"

    @field_validator("message")
    @classmethod
    def validate_message(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("问题不能为空")
        return v


class ChatResponse(BaseModel):
    session_id: str
    content: str
    intent: str = ""