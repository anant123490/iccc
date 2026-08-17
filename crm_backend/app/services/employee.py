import logging
from secrets import token_urlsafe

from fastapi import HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.password import hash_password
from app.dependencies.pagination import PaginationParams
from app.models.enums import UserRole
from app.models.user import User
from app.repositories.user import UserRepository
from app.schemas.common import PageMeta
from app.schemas.employee import (
    EmployeeInviteRequest,
    EmployeeInviteResponse,
    EmployeeListResponse,
    EmployeeRoleUpdate,
    EmployeeUpdate,
)
from app.schemas.user import UserResponse

logger = logging.getLogger(__name__)


class EmployeeService:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db
        self.users = UserRepository(db)

    async def invite_employee(
        self,
        *,
        current_user: User,
        data: EmployeeInviteRequest,
    ) -> EmployeeInviteResponse:
        self._validate_invited_role(current_user, data.role)

        existing_user = await self.users.get_by_email(str(data.email))
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="A user with this email already exists.",
            )

        temporary_password = self._generate_temporary_password()
        employee = await self.users.create(
            organization_id=current_user.organization_id,
            email=str(data.email),
            full_name=data.full_name,
            hashed_password=hash_password(temporary_password),
            role=data.role,
        )

        await self.db.commit()
        logger.info(
            "Mock invite sent to user id=%s in organization id=%s",
            employee.id,
            current_user.organization_id,
        )

        return EmployeeInviteResponse(
            employee=UserResponse.model_validate(employee),
            temporary_password=temporary_password,
            message="Mock invite created. Share the temporary password manually.",
        )

    async def list_employees(
        self,
        *,
        current_user: User,
        pagination: PaginationParams,
        search: str | None = None,
        role: UserRole | None = None,
        is_active: bool | None = None,
    ) -> EmployeeListResponse:
        employees, total = await self.users.list_by_organization(
            organization_id=current_user.organization_id,
            offset=pagination.offset,
            limit=pagination.size,
            search=search,
            role=role,
            is_active=is_active,
        )

        return EmployeeListResponse(
            items=[UserResponse.model_validate(employee) for employee in employees],
            meta=PageMeta(page=pagination.page, size=pagination.size, total=total),
        )

    async def get_employee(self, *, current_user: User, employee_id: int) -> UserResponse:
        employee = await self._get_employee_or_404(current_user, employee_id)
        return UserResponse.model_validate(employee)

    async def update_employee(
        self,
        *,
        current_user: User,
        employee_id: int,
        data: EmployeeUpdate,
    ) -> UserResponse:
        employee = await self._get_employee_or_404(current_user, employee_id)
        self._validate_target_can_be_managed(current_user, employee)

        employee = await self.users.update(employee, data)
        await self.db.commit()
        return UserResponse.model_validate(employee)

    async def assign_role(
        self,
        *,
        current_user: User,
        employee_id: int,
        data: EmployeeRoleUpdate,
    ) -> UserResponse:
        if current_user.role != UserRole.OWNER:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only owners can assign employee roles.",
            )

        if data.role == UserRole.OWNER:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Owner transfer is not supported in this module.",
            )

        employee = await self._get_employee_or_404(current_user, employee_id)
        if employee.role == UserRole.OWNER:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Owner role cannot be changed from this endpoint.",
            )

        employee = await self.users.assign_role(employee, data.role)
        await self.db.commit()
        return UserResponse.model_validate(employee)

    async def disable_employee(
        self,
        *,
        current_user: User,
        employee_id: int,
    ) -> UserResponse:
        employee = await self._get_employee_or_404(current_user, employee_id)

        if employee.id == current_user.id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="You cannot disable your own account.",
            )

        if employee.role == UserRole.OWNER:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Owner accounts cannot be disabled from this endpoint.",
            )

        employee = await self.users.disable(employee)
        await self.db.commit()
        return UserResponse.model_validate(employee)

    async def _get_employee_or_404(self, current_user: User, employee_id: int) -> User:
        employee = await self.users.get_by_id(
            user_id=employee_id,
            organization_id=current_user.organization_id,
        )
        if not employee:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Employee not found.",
            )
        return employee

    def _validate_invited_role(self, current_user: User, role: UserRole) -> None:
        if role == UserRole.OWNER:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Owner role cannot be invited through employee management.",
            )

        if current_user.role == UserRole.ADMIN and role != UserRole.EMPLOYEE:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admins can only invite employees.",
            )

    def _validate_target_can_be_managed(
        self,
        current_user: User,
        employee: User,
    ) -> None:
        if employee.role == UserRole.OWNER and employee.id != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only the owner can update the owner account.",
            )

    def _generate_temporary_password(self) -> str:
        return f"Temp-{token_urlsafe(12)}"
