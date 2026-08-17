from typing import Generic, TypeVar

from pydantic import BaseModel, ConfigDict

T = TypeVar("T")


class APIError(BaseModel):
    detail: str


class HealthResponse(BaseModel):
    status: str
    app_name: str
    version: str
    environment: str


class PageMeta(BaseModel):
    page: int
    size: int
    total: int


class PaginatedResponse(BaseModel, Generic[T]):
    model_config = ConfigDict(from_attributes=True)

    items: list[T]
    meta: PageMeta
