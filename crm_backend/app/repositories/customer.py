from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.customer import Customer
from app.models.enums import CustomerStatus
from app.schemas.customer import CustomerCreate, CustomerUpdate


class CustomerRepository:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def get_by_id(
        self,
        *,
        customer_id: int,
        organization_id: int,
    ) -> Customer | None:
        result = await self.db.execute(
            select(Customer).where(
                Customer.id == customer_id,
                Customer.organization_id == organization_id,
            )
        )
        return result.scalar_one_or_none()

    async def get_by_email(
        self,
        *,
        organization_id: int,
        email: str,
    ) -> Customer | None:
        result = await self.db.execute(
            select(Customer).where(
                Customer.organization_id == organization_id,
                func.lower(Customer.email) == email.lower(),
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
        status: CustomerStatus | None = None,
        assigned_employee_id: int | None = None,
    ) -> tuple[list[Customer], int]:
        filters = [Customer.organization_id == organization_id]

        if search:
            search_text = f"%{search.lower()}%"
            filters.append(
                or_(
                    func.lower(Customer.name).like(search_text),
                    func.lower(Customer.email).like(search_text),
                    func.lower(Customer.company).like(search_text),
                    func.lower(Customer.phone).like(search_text),
                )
            )

        if status:
            filters.append(Customer.status == status)

        if assigned_employee_id:
            filters.append(Customer.assigned_employee_id == assigned_employee_id)

        total_result = await self.db.execute(
            select(func.count()).select_from(Customer).where(*filters)
        )
        total = total_result.scalar_one()

        customers_result = await self.db.execute(
            select(Customer)
            .where(*filters)
            .order_by(Customer.created_at.desc(), Customer.id.desc())
            .offset(offset)
            .limit(limit)
        )
        return list(customers_result.scalars().all()), total

    async def create(
        self,
        *,
        organization_id: int,
        data: CustomerCreate,
    ) -> Customer:
        customer_data = data.model_dump()
        if customer_data.get("email"):
            customer_data["email"] = customer_data["email"].lower()

        customer = Customer(organization_id=organization_id, **customer_data)
        self.db.add(customer)
        await self.db.flush()
        return customer

    async def update(self, customer: Customer, data: CustomerUpdate) -> Customer:
        update_data = data.model_dump(exclude_unset=True)
        if update_data.get("email"):
            update_data["email"] = update_data["email"].lower()

        for field, value in update_data.items():
            setattr(customer, field, value)

        self.db.add(customer)
        await self.db.flush()
        return customer

    async def delete(self, customer: Customer) -> None:
        await self.db.delete(customer)
        await self.db.flush()
