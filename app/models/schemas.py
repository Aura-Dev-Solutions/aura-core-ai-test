from pydantic import BaseModel
from typing import List


class Entity(BaseModel):
    text: str
    label: str


class DocumentResponse(BaseModel):
    text: str
    embeddings: List[float]
    category: str
    entities: List[Entity]
