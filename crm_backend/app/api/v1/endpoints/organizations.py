from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies.auth import get_current_user, require_roles
from app.dependencies.database import get_db
from app.models.enums import UserRole
from app.models.user import User
from app.schemas.organization import OrganizationResponse, OrganizationUpdate
from app.services.organization import OrganizationService

router = APIRouter(prefix="/organizations", tags=["Organizations"])


@router.get("/me", response_model=OrganizationResponse)
async def get_my_organization(
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> OrganizationResponse:
    return await OrganizationService(db).get_current_organization(current_user)


@router.patch("/me", response_model=OrganizationResponse)
async def update_my_organization(
    data: OrganizationUpdate,
    current_user: Annotated[
        User,
        Depends(require_roles(UserRole.OWNER, UserRole.ADMIN)),
    ],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> OrganizationResponse:
    return await OrganizationService(db).update_current_organization(
        current_user,
        data,
    )
