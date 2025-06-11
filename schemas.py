from pydantic import BaseModel
from datetime import datetime
from typing import List

class MessageCreate(BaseModel):
    content: str

class MessageResponse(BaseModel):
    content: str
    role: str
    timestamp: datetime

    class Config:
        from_attributes = True

class ChatResponse(BaseModel):
    id: int
    created_at: datetime
    messages: List[MessageResponse] = []

    class Config:
        from_attributes = True