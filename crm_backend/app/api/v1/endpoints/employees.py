from typing import Annotated

from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies.auth import require_roles
from app.dependencies.database import get_db
from app.dependencies.pagination import PaginationParams, get_pagination_params
from app.models.enums import UserRole
from app.models.user import User
from app.schemas.employee import (
    EmployeeInviteRequest,
    EmployeeInviteResponse,
    EmployeeListResponse,
    EmployeeRoleUpdate,
    EmployeeUpdate,
)
from app.schemas.user import UserResponse
from app.services.employee import EmployeeService

router = APIRouter(prefix="/employees", tags=["Employees"])

ManagerUser = Annotated[
    User,
    Depends(require_roles(UserRole.OWNER, UserRole.ADMIN)),
]


@router.post(
    "/invite",
    response_model=EmployeeInviteResponse,
    status_code=status.HTTP_201_CREATED,
)
async def invite_employee(
    data: EmployeeInviteRequest,
    current_user: ManagerUser,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> EmployeeInviteResponse:
    return await EmployeeService(db).invite_employee(
        current_user=current_user,
        data=data,
    )


@router.get("", response_model=EmployeeListResponse)
async def list_employees(
    current_user: ManagerUser,
    db: Annotated[AsyncSession, Depends(get_db)],
    pagination: Annotated[PaginationParams, Depends(get_pagination_params)],
    search: str | None = Query(default=None, min_length=1, max_length=100),
    role: UserRole | None = None,
    is_active: bool | None = None,
) -> EmployeeListResponse:
    return await EmployeeService(db).list_employees(
        current_user=current_user,
        pagination=pagination,
        search=search,
        role=role,
        is_active=is_active,
    )


@router.get("/{employee_id}", response_model=UserResponse)
async def get_employee(
    employee_id: int,
    current_user: ManagerUser,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> UserResponse:
    return await EmployeeService(db).get_employee(
        current_user=current_user,
        employee_id=employee_id,
    )


@router.patch("/{employee_id}", response_model=UserResponse)
async def update_employee(
    employee_id: int,
    data: EmployeeUpdate,
    current_user: ManagerUser,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> UserResponse:
    return await EmployeeService(db).update_employee(
        current_user=current_user,
        employee_id=employee_id,
        data=data,
    )


@router.patch("/{employee_id}/role", response_model=UserResponse)
async def assign_employee_role(
    employee_id: int,
    data: EmployeeRoleUpdate,
    current_user: ManagerUser,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> UserResponse:
    return await EmployeeService(db).assign_role(
        current_user=current_user,
        employee_id=employee_id,
        data=data,
    )


@router.patch("/{employee_id}/disable", response_model=UserResponse)
async def disable_employee(
    employee_id: int,
    current_user: ManagerUser,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> UserResponse:
    return await EmployeeService(db).disable_employee(
        current_user=current_user,
        employee_id=employee_id,
    )
