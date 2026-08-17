from typing import Annotated

from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies.auth import get_current_user, require_roles
from app.dependencies.database import get_db
from app.dependencies.pagination import PaginationParams, get_pagination_params
from app.models.enums import CustomerStatus, UserRole
from app.models.user import User
from app.schemas.auth import MessageResponse
from app.schemas.customer import (
    CustomerCreate,
    CustomerListResponse,
    CustomerResponse,
    CustomerUpdate,
)
from app.services.customer import CustomerService

router = APIRouter(prefix="/customers", tags=["Customers"])

AuthenticatedUser = Annotated[User, Depends(get_current_user)]
ManagerUser = Annotated[
    User,
    Depends(require_roles(UserRole.OWNER, UserRole.ADMIN)),
]


@router.post("", response_model=CustomerResponse, status_code=status.HTTP_201_CREATED)
async def create_customer(
    data: CustomerCreate,
    current_user: AuthenticatedUser,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> CustomerResponse:
    return await CustomerService(db).create_customer(
        current_user=current_user,
        data=data,
    )


@router.get("", response_model=CustomerListResponse)
async def list_customers(
    current_user: AuthenticatedUser,
    db: Annotated[AsyncSession, Depends(get_db)],
    pagination: Annotated[PaginationParams, Depends(get_pagination_params)],
    search: str | None = Query(default=None, min_length=1, max_length=100),
    status: CustomerStatus | None = None,
    assigned_employee_id: int | None = None,
) -> CustomerListResponse:
    return await CustomerService(db).list_customers(
        current_user=current_user,
        pagination=pagination,
        search=search,
        status=status,
        assigned_employee_id=assigned_employee_id,
    )


@router.get("/{customer_id}", response_model=CustomerResponse)
async def get_customer(
    customer_id: int,
    current_user: AuthenticatedUser,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> CustomerResponse:
    return await CustomerService(db).get_customer(
        current_user=current_user,
        customer_id=customer_id,
    )


@router.patch("/{customer_id}", response_model=CustomerResponse)
async def update_customer(
    customer_id: int,
    data: CustomerUpdate,
    current_user: AuthenticatedUser,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> CustomerResponse:
    return await CustomerService(db).update_customer(
        current_user=current_user,
        customer_id=customer_id,
        data=data,
    )


@router.delete("/{customer_id}", response_model=MessageResponse)
async def delete_customer(
    customer_id: int,
    current_user: ManagerUser,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> MessageResponse:
    await CustomerService(db).delete_customer(
        current_user=current_user,
        customer_id=customer_id,
    )
    return MessageResponse(message="Customer deleted successfully.")
