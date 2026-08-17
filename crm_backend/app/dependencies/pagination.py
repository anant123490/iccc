from fastapi import Query
from pydantic import BaseModel


class PaginationParams(BaseModel):
    page: int
    size: int

    @property
    def offset(self) -> int:
        return (self.page - 1) * self.size


def get_pagination_params(
    page: int = Query(1, ge=1),
    size: int = Query(20, ge=1, le=100),
) -> PaginationParams:
    return PaginationParams(page=page, size=size)
