import logging

from fastapi import HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.jwt import (
    InvalidTokenError,
    create_access_token,
    create_refresh_token,
    decode_token,
)
from app.auth.password import hash_password, verify_password
from app.auth.token_hash import hash_token
from app.models.enums import UserRole
from app.models.user import User
from app.repositories.organization import OrganizationRepository
from app.repositories.refresh_token import RefreshTokenRepository
from app.repositories.user import UserRepository
from app.schemas.auth import (
    AuthResponse,
    ForgotPasswordRequest,
    LoginRequest,
    RegisterOrganizationRequest,
    TokenResponse,
)
from app.schemas.organization import OrganizationResponse
from app.schemas.user import UserResponse

logger = logging.getLogger(__name__)


class AuthService:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db
        self.organizations = OrganizationRepository(db)
        self.users = UserRepository(db)
        self.refresh_tokens = RefreshTokenRepository(db)

    async def register_organization(
        self,
        data: RegisterOrganizationRequest,
    ) -> AuthResponse:
        existing_organization = await self.organizations.get_by_slug(
            data.organization.slug
        )
        if existing_organization:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Organization slug is already taken.",
            )

        existing_user = await self.users.get_by_email(str(data.owner_email))
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="A user with this email already exists.",
            )

        organization = await self.organizations.create(
            name=data.organization.name,
            slug=data.organization.slug,
            settings=data.organization.settings,
        )
        owner = await self.users.create(
            organization_id=organization.id,
            email=str(data.owner_email),
            full_name=data.owner_full_name,
            hashed_password=hash_password(data.owner_password.get_secret_value()),
            role=UserRole.OWNER,
        )
        token_response = await self._create_and_store_token_pair(owner)

        await self.db.commit()
        logger.info("Registered organization id=%s", organization.id)

        return AuthResponse(
            **token_response.model_dump(),
            user=UserResponse.model_validate(owner),
            organization=OrganizationResponse.model_validate(organization),
        )

    async def login(self, data: LoginRequest) -> AuthResponse:
        user = await self.users.get_by_email(str(data.email))
        if not user or not verify_password(
            data.password.get_secret_value(),
            user.hashed_password,
        ):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect email or password.",
            )

        if not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="This user account is disabled.",
            )

        organization = await self.organizations.get_by_id(user.organization_id)
        if not organization or not organization.is_active:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="This organization is disabled.",
            )

        token_response = await self._create_and_store_token_pair(user)
        await self.db.commit()
        logger.info("User id=%s logged in", user.id)

        return AuthResponse(
            **token_response.model_dump(),
            user=UserResponse.model_validate(user),
            organization=OrganizationResponse.model_validate(organization),
        )

    async def refresh(self, refresh_token: str) -> TokenResponse:
        payload = self._decode_refresh_token(refresh_token)
        stored_token = await self.refresh_tokens.get_active_by_hash(
            hash_token(refresh_token)
        )

        if not stored_token or stored_token.jti != payload.get("jti"):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid refresh token.",
            )

        user = await self.users.get_by_id(
            user_id=int(payload["sub"]),
            organization_id=int(payload["organization_id"]),
        )
        if not user or not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid refresh token.",
            )

        await self.refresh_tokens.revoke(stored_token)
        token_response = await self._create_and_store_token_pair(user)
        await self.db.commit()
        return token_response

    async def logout(self, refresh_token: str) -> None:
        stored_token = await self.refresh_tokens.get_active_by_hash(
            hash_token(refresh_token)
        )
        if stored_token:
            await self.refresh_tokens.revoke(stored_token)
            await self.db.commit()

    async def forgot_password(self, data: ForgotPasswordRequest) -> None:
        user = await self.users.get_by_email(str(data.email))
        if user:
            logger.info("Mock password reset requested for user id=%s", user.id)

    async def verify_email(self, current_user: User) -> User:
        user = await self.users.mark_verified(current_user)
        await self.db.commit()
        logger.info("Mock email verification completed for user id=%s", user.id)
        return user

    async def _create_and_store_token_pair(self, user: User) -> TokenResponse:
        access_token = create_access_token(
            user_id=user.id,
            organization_id=user.organization_id,
            role=user.role,
        )
        refresh_token, refresh_expires_at, refresh_jti = create_refresh_token(
            user_id=user.id,
            organization_id=user.organization_id,
            role=user.role,
        )
        await self.refresh_tokens.create(
            organization_id=user.organization_id,
            user_id=user.id,
            jti=refresh_jti,
            token_hash=hash_token(refresh_token),
            expires_at=refresh_expires_at,
        )
        return TokenResponse(access_token=access_token, refresh_token=refresh_token)

    def _decode_refresh_token(self, refresh_token: str) -> dict:
        try:
            return decode_token(refresh_token, expected_type="refresh")
        except InvalidTokenError as exc:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid refresh token.",
            ) from exc
