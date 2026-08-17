import pytest
from fastapi import HTTPException

from app.models.enums import UserRole
from app.models.user import User
from app.services.employee import EmployeeService


def make_user(*, role: UserRole) -> User:
    return User(
        id=1,
        organization_id=10,
        email="manager@example.com",
        full_name="Test Manager",
        hashed_password="hashed",
        role=role,
    )


def test_admin_can_only_invite_employee_role() -> None:
    service = EmployeeService(db=None)
    admin = make_user(role=UserRole.ADMIN)

    service._validate_invited_role(admin, UserRole.EMPLOYEE)

    with pytest.raises(HTTPException) as exc:
        service._validate_invited_role(admin, UserRole.ADMIN)

    assert exc.value.status_code == 403


def test_owner_role_cannot_be_invited() -> None:
    service = EmployeeService(db=None)
    owner = make_user(role=UserRole.OWNER)

    with pytest.raises(HTTPException) as exc:
        service._validate_invited_role(owner, UserRole.OWNER)

    assert exc.value.status_code == 400


def test_temporary_password_is_generated_for_mock_invite() -> None:
    service = EmployeeService(db=None)

    temporary_password = service._generate_temporary_password()

    assert temporary_password.startswith("Temp-")
    assert len(temporary_password) > 12
