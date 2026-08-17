from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.organization import Organization
from app.schemas.organization import OrganizationUpdate


class OrganizationRepository:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def get_by_id(self, organization_id: int) -> Organization | None:
        result = await self.db.execute(
            select(Organization).where(Organization.id == organization_id)
        )
        return result.scalar_one_or_none()

    async def get_by_slug(self, slug: str) -> Organization | None:
        result = await self.db.execute(select(Organization).where(Organization.slug == slug))
        return result.scalar_one_or_none()

    async def create(
        self,
        *,
        name: str,
        slug: str,
        settings: dict,
    ) -> Organization:
        organization = Organization(name=name, slug=slug, settings=settings)
        self.db.add(organization)
        await self.db.flush()
        return organization

    async def update(
        self,
        organization: Organization,
        data: OrganizationUpdate,
    ) -> Organization:
        update_data = data.model_dump(exclude_unset=True)
        for field, value in update_data.items():
            setattr(organization, field, value)

        self.db.add(organization)
        await self.db.flush()
        return organization
