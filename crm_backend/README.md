# Multi-Tenant SaaS CRM Backend

Production-inspired, student-friendly CRM backend built with FastAPI, PostgreSQL,
SQLAlchemy 2.0 async ORM, Alembic, JWT auth, and role based access control.

This project is intentionally built one module at a time so every file stays
easy to explain in interviews.

## Current Module

Module 3: customer management.

Included so far:

- FastAPI app factory
- Environment based settings
- Structured logging setup
- Async SQLAlchemy engine and session dependency
- Alembic async migration setup
- Health check endpoint
- Docker support
- Basic test
- Organization model and current-organization APIs
- User model with owner/admin/employee roles
- JWT access and refresh token flow
- Hashed refresh token storage for logout and rotation
- Mock forgot password and email verification endpoints
- Employee invite flow with mock temporary passwords
- Employee listing with pagination, role filtering, active filtering, and search
- Employee profile updates, role assignment, and disable action
- Customer CRUD with tenant isolation
- Customer search, pagination, status filter, and assigned employee filter
- Customer assignment validation against active users in the same organization

## Module 1 Endpoints

| Method | Endpoint | Description |
| --- | --- | --- |
| POST | `/api/v1/auth/register-organization` | Create an organization and owner user |
| POST | `/api/v1/auth/login` | Login with email and password |
| POST | `/api/v1/auth/refresh` | Rotate refresh token and issue new tokens |
| POST | `/api/v1/auth/logout` | Revoke a refresh token |
| POST | `/api/v1/auth/forgot-password` | Mock password reset request |
| POST | `/api/v1/auth/verify-email` | Mock email verification for logged-in user |
| GET | `/api/v1/auth/me` | Return the logged-in user |
| GET | `/api/v1/organizations/me` | Return the current tenant organization |
| PATCH | `/api/v1/organizations/me` | Update organization name/settings as owner/admin |

## Module 2 Endpoints

| Method | Endpoint | Description |
| --- | --- | --- |
| POST | `/api/v1/employees/invite` | Invite an employee or admin inside the current organization |
| GET | `/api/v1/employees` | List employees with pagination, search, and filters |
| GET | `/api/v1/employees/{employee_id}` | Get one employee from the current organization |
| PATCH | `/api/v1/employees/{employee_id}` | Update employee profile fields |
| PATCH | `/api/v1/employees/{employee_id}/role` | Assign admin/employee role as owner |
| PATCH | `/api/v1/employees/{employee_id}/disable` | Disable an employee account |

## Module 3 Endpoints

| Method | Endpoint | Description |
| --- | --- | --- |
| POST | `/api/v1/customers` | Create a customer in the current organization |
| GET | `/api/v1/customers` | List customers with pagination, search, and filters |
| GET | `/api/v1/customers/{customer_id}` | Get one customer from the current organization |
| PATCH | `/api/v1/customers/{customer_id}` | Update customer details or assignment |
| DELETE | `/api/v1/customers/{customer_id}` | Delete a customer as owner/admin |

## Local Setup

```bash
cd crm_backend
copy .env.example .env
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

API docs:

- Swagger UI: http://localhost:8000/docs
- OpenAPI JSON: http://localhost:8000/api/v1/openapi.json

## Docker Setup

```bash
cd crm_backend
docker compose up --build
```

## Tests

```bash
cd crm_backend
pytest
```
