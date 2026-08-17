from app.models import Customer, Organization, RefreshToken, User  # noqa: F401
from app.models.base import Base

# Alembic reads this metadata when it creates migration files.
# Importing models above makes sure Alembic can detect their tables.
target_metadata = Base.metadata
