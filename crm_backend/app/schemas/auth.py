from pydantic import BaseModel, EmailStr, Field, SecretStr

from app.schemas.organization import OrganizationCreate, OrganizationResponse
from app.schemas.user import UserResponse


class RegisterOrganizationRequest(BaseModel):
    organization: OrganizationCreate
    owner_full_name: str = Field(min_length=2, max_length=120)
    owner_email: EmailStr
    owner_password: SecretStr = Field(min_length=8, max_length=128)


class LoginRequest(BaseModel):
    email: EmailStr
    password: SecretStr


class RefreshTokenRequest(BaseModel):
    refresh_token: str


class ForgotPasswordRequest(BaseModel):
    email: EmailStr


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class AuthResponse(TokenResponse):
    user: UserResponse
    organization: OrganizationResponse


class MessageResponse(BaseModel):
    message: str
