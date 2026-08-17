from enum import Enum


class UserRole(str, Enum):
    OWNER = "owner"
    ADMIN = "admin"
    EMPLOYEE = "employee"


class CustomerStatus(str, Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
