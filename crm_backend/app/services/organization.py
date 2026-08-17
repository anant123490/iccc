from fastapi import HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.user import User
from app.repositories.organization import OrganizationRepository
from app.schemas.organization import OrganizationResponse, OrganizationUpdate


class OrganizationService:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db
        self.organizations = OrganizationRepository(db)

    async def get_current_organization(self, current_user: User) -> OrganizationResponse:
        organization = await self.organizations.get_by_id(current_user.organization_id)
        if not organization:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Organization not found.",
            )

        return OrganizationResponse.model_validate(organization)

    async def update_current_organization(
        self,
        current_user: User,
        data: OrganizationUpdate,
    ) -> OrganizationResponse:
        organization = await self.organizations.get_by_id(current_user.organization_id)
        if not organization:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Organization not found.",
            )

        organization = await self.organizations.update(organization, data)
        await self.db.commit()
        return OrganizationResponse.model_validate(organization)
