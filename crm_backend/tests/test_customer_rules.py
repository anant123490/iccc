import pytest
from fastapi import HTTPException

from app.models.customer import Customer
from app.models.enums import UserRole
from app.models.user import User
from app.services.customer import CustomerService


class FakeUserRepository:
    def __init__(self, user: User | None) -> None:
        self.user = user

    async def get_by_id(self, *, user_id: int, organization_id: int) -> User | None:
        if (
            self.user
            and self.user.id == user_id
            and self.user.organization_id == organization_id
        ):
            return self.user
        return None


class FakeCustomerRepository:
    def __init__(self, customer: Customer | None) -> None:
        self.customer = customer

    async def get_by_email(self, *, organization_id: int, email: str) -> Customer | None:
        if (
            self.customer
            and self.customer.organization_id == organization_id
            and self.customer.email == email
        ):
            return self.customer
        return None


def make_user(*, user_id: int = 5, organization_id: int = 10) -> User:
    return User(
        id=user_id,
        organization_id=organization_id,
        email="employee@example.com",
        full_name="Test Employee",
        hashed_password="hashed",
        role=UserRole.EMPLOYEE,
        is_active=True,
    )


@pytest.mark.asyncio
async def test_assigned_employee_must_belong_to_same_organization() -> None:
    service = CustomerService(db=None)
    service.users = FakeUserRepository(make_user(organization_id=20))

    with pytest.raises(HTTPException) as exc:
        await service._validate_assigned_employee(
            organization_id=10,
            assigned_employee_id=5,
        )

    assert exc.value.status_code == 400


@pytest.mark.asyncio
async def test_active_same_org_employee_can_be_assigned() -> None:
    service = CustomerService(db=None)
    service.users = FakeUserRepository(make_user(organization_id=10))

    await service._validate_assigned_employee(
        organization_id=10,
        assigned_employee_id=5,
    )


@pytest.mark.asyncio
async def test_customer_email_conflict_is_checked_inside_tenant() -> None:
    service = CustomerService(db=None)
    service.customers = FakeCustomerRepository(
        Customer(id=1, organization_id=10, name="Existing", email="a@example.com")
    )

    with pytest.raises(HTTPException) as exc:
        await service._ensure_email_available(
            organization_id=10,
            email="a@example.com",
        )

    assert exc.value.status_code == 409


@pytest.mark.asyncio
async def test_same_email_in_different_tenant_is_allowed() -> None:
    service = CustomerService(db=None)
    service.customers = FakeCustomerRepository(
        Customer(id=1, organization_id=20, name="Existing", email="a@example.com")
    )

    await service._ensure_email_available(
        organization_id=10,
        email="a@example.com",
    )
