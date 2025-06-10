from pydantic import BaseModel
from datetime import datetime

class MessageCreate(BaseModel):
    content: str

class MessageResponse(BaseModel):
    content: str
    role: str
    timestamp: datetime

    class Config:
        orm_mode = True