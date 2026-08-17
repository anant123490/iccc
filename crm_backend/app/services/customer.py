from fastapi import HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies.pagination import PaginationParams
from app.models.customer import Customer
from app.models.enums import CustomerStatus
from app.models.user import User
from app.repositories.customer import CustomerRepository
from app.repositories.user import UserRepository
from app.schemas.common import PageMeta
from app.schemas.customer import (
    CustomerCreate,
    CustomerListResponse,
    CustomerResponse,
    CustomerUpdate,
)


class CustomerService:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db
        self.customers = CustomerRepository(db)
        self.users = UserRepository(db)

    async def create_customer(
        self,
        *,
        current_user: User,
        data: CustomerCreate,
    ) -> CustomerResponse:
        await self._validate_assigned_employee(
            organization_id=current_user.organization_id,
            assigned_employee_id=data.assigned_employee_id,
        )
        await self._ensure_email_available(
            organization_id=current_user.organization_id,
            email=str(data.email) if data.email else None,
        )

        customer = await self.customers.create(
            organization_id=current_user.organization_id,
            data=data,
        )
        await self.db.commit()
        return CustomerResponse.model_validate(customer)

    async def list_customers(
        self,
        *,
        current_user: User,
        pagination: PaginationParams,
        search: str | None = None,
        status: CustomerStatus | None = None,
        assigned_employee_id: int | None = None,
    ) -> CustomerListResponse:
        if assigned_employee_id:
            await self._validate_assigned_employee(
                organization_id=current_user.organization_id,
                assigned_employee_id=assigned_employee_id,
            )

        customers, total = await self.customers.list_by_organization(
            organization_id=current_user.organization_id,
            offset=pagination.offset,
            limit=pagination.size,
            search=search,
            status=status,
            assigned_employee_id=assigned_employee_id,
        )

        return CustomerListResponse(
            items=[CustomerResponse.model_validate(customer) for customer in customers],
            meta=PageMeta(page=pagination.page, size=pagination.size, total=total),
        )

    async def get_customer(
        self,
        *,
        current_user: User,
        customer_id: int,
    ) -> CustomerResponse:
        customer = await self._get_customer_or_404(current_user, customer_id)
        return CustomerResponse.model_validate(customer)

    async def update_customer(
        self,
        *,
        current_user: User,
        customer_id: int,
        data: CustomerUpdate,
    ) -> CustomerResponse:
        customer = await self._get_customer_or_404(current_user, customer_id)

        if "assigned_employee_id" in data.model_fields_set:
            await self._validate_assigned_employee(
                organization_id=current_user.organization_id,
                assigned_employee_id=data.assigned_employee_id,
            )

        if "email" in data.model_fields_set:
            await self._ensure_email_available(
                organization_id=current_user.organization_id,
                email=str(data.email) if data.email else None,
                current_customer_id=customer.id,
            )

        customer = await self.customers.update(customer, data)
        await self.db.commit()
        return CustomerResponse.model_validate(customer)

    async def delete_customer(
        self,
        *,
        current_user: User,
        customer_id: int,
    ) -> None:
        customer = await self._get_customer_or_404(current_user, customer_id)
        await self.customers.delete(customer)
        await self.db.commit()

    async def _get_customer_or_404(
        self,
        current_user: User,
        customer_id: int,
    ) -> Customer:
        customer = await self.customers.get_by_id(
            customer_id=customer_id,
            organization_id=current_user.organization_id,
        )
        if not customer:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Customer not found.",
            )
        return customer

    async def _validate_assigned_employee(
        self,
        *,
        organization_id: int,
        assigned_employee_id: int | None,
    ) -> None:
        if assigned_employee_id is None:
            return

        employee = await self.users.get_by_id(
            user_id=assigned_employee_id,
            organization_id=organization_id,
        )
        if not employee or not employee.is_active:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Assigned employee must be an active user in this organization.",
            )

    async def _ensure_email_available(
        self,
        *,
        organization_id: int,
        email: str | None,
        current_customer_id: int | None = None,
    ) -> None:
        if not email:
            return

        existing_customer = await self.customers.get_by_email(
            organization_id=organization_id,
            email=email,
        )
        if existing_customer and existing_customer.id != current_customer_id:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="A customer with this email already exists.",
            )
