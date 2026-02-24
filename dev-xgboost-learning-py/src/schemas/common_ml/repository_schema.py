from pydantic import BaseModel
from datetime import datetime


class ModelRepositoryResponse(BaseModel):
    model_id: str
    filename: str
    size_kb: float
    created_at: datetime

