from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class OrganizationBase(BaseModel):
    name: str = Field(min_length=2, max_length=120)


class OrganizationCreate(OrganizationBase):
    slug: str = Field(
        min_length=3,
        max_length=80,
        pattern=r"^[a-z0-9-]+$",
        description="Unique URL-friendly organization name, like acme-crm.",
    )
    settings: dict[str, Any] = Field(default_factory=dict)


class OrganizationUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=2, max_length=120)
    settings: dict[str, Any] | None = None


class OrganizationResponse(OrganizationBase):
    model_config = ConfigDict(from_attributes=True)

    id: int
    slug: str
    settings: dict[str, Any]
    is_active: bool
