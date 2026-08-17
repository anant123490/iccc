from pydantic import BaseModel, EmailStr, Field

from app.models.enums import UserRole
from app.schemas.common import PaginatedResponse
from app.schemas.user import UserResponse


class EmployeeInviteRequest(BaseModel):
    email: EmailStr
    full_name: str = Field(min_length=2, max_length=120)
    role: UserRole = UserRole.EMPLOYEE


class EmployeeInviteResponse(BaseModel):
    employee: UserResponse
    temporary_password: str
    message: str


class EmployeeUpdate(BaseModel):
    full_name: str | None = Field(default=None, min_length=2, max_length=120)


class EmployeeRoleUpdate(BaseModel):
    role: UserRole


class EmployeeListResponse(PaginatedResponse[UserResponse]):
    pass
