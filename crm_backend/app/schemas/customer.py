from pydantic import BaseModel, ConfigDict, EmailStr, Field

from app.models.enums import CustomerStatus
from app.schemas.common import PaginatedResponse


class CustomerBase(BaseModel):
    name: str = Field(min_length=2, max_length=120)
    email: EmailStr | None = None
    phone: str | None = Field(default=None, max_length=30)
    company: str | None = Field(default=None, max_length=120)
    status: CustomerStatus = CustomerStatus.ACTIVE
    assigned_employee_id: int | None = None
    notes: str | None = Field(default=None, max_length=2000)


class CustomerCreate(CustomerBase):
    pass


class CustomerUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=2, max_length=120)
    email: EmailStr | None = None
    phone: str | None = Field(default=None, max_length=30)
    company: str | None = Field(default=None, max_length=120)
    status: CustomerStatus | None = None
    assigned_employee_id: int | None = None
    notes: str | None = Field(default=None, max_length=2000)


class CustomerResponse(CustomerBase):
    model_config = ConfigDict(from_attributes=True)

    id: int
    organization_id: int


class CustomerListResponse(PaginatedResponse[CustomerResponse]):
    pass
