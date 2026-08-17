from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.enums import UserRole
from app.models.user import User
from app.schemas.employee import EmployeeUpdate


class UserRepository:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def get_by_email(self, email: str) -> User | None:
        result = await self.db.execute(select(User).where(User.email == email.lower()))
        return result.scalar_one_or_none()

    async def get_by_id(self, *, user_id: int, organization_id: int) -> User | None:
        result = await self.db.execute(
            select(User).where(
                User.id == user_id,
                User.organization_id == organization_id,
            )
        )
        return result.scalar_one_or_none()

    async def list_by_organization(
        self,
        *,
        organization_id: int,
        offset: int,
        limit: int,
        search: str | None = None,
        role: UserRole | None = None,
        is_active: bool | None = None,
    ) -> tuple[list[User], int]:
        filters = [User.organization_id == organization_id]

        if search:
            search_text = f"%{search.lower()}%"
            filters.append(
                (func.lower(User.full_name).like(search_text))
                | (func.lower(User.email).like(search_text))
            )

        if role:
            filters.append(User.role == role)

        if is_active is not None:
            filters.append(User.is_active == is_active)

        total_result = await self.db.execute(
            select(func.count()).select_from(User).where(*filters)
        )
        total = total_result.scalar_one()

        users_result = await self.db.execute(
            select(User)
            .where(*filters)
            .order_by(User.created_at.desc(), User.id.desc())
            .offset(offset)
            .limit(limit)
        )
        return list(users_result.scalars().all()), total

    async def create(
        self,
        *,
        organization_id: int,
        email: str,
        full_name: str,
        hashed_password: str,
        role: UserRole,
    ) -> User:
        user = User(
            organization_id=organization_id,
            email=email.lower(),
            full_name=full_name,
            hashed_password=hashed_password,
            role=role,
        )
        self.db.add(user)
        await self.db.flush()
        return user

    async def update(self, user: User, data: EmployeeUpdate) -> User:
        update_data = data.model_dump(exclude_unset=True)
        for field, value in update_data.items():
            setattr(user, field, value)

        self.db.add(user)
        await self.db.flush()
        return user

    async def assign_role(self, user: User, role: UserRole) -> User:
        user.role = role
        self.db.add(user)
        await self.db.flush()
        return user

    async def disable(self, user: User) -> User:
        user.is_active = False
        self.db.add(user)
        await self.db.flush()
        return user

    async def mark_verified(self, user: User) -> User:
        user.is_verified = True
        self.db.add(user)
        await self.db.flush()
        return user
