from datetime import UTC, datetime, timedelta
from uuid import uuid4

from jose import JWTError, jwt

from app.core.config import settings
from app.models.enums import UserRole


class InvalidTokenError(Exception):
    pass


def _create_token(
    *,
    user_id: int,
    organization_id: int,
    role: UserRole,
    token_type: str,
    expires_delta: timedelta,
    jti: str | None = None,
) -> tuple[str, datetime, str]:
    expires_at = datetime.now(UTC) + expires_delta
    token_jti = jti or str(uuid4())
    payload = {
        "sub": str(user_id),
        "organization_id": organization_id,
        "role": role.value,
        "type": token_type,
        "jti": token_jti,
        "iat": datetime.now(UTC),
        "exp": expires_at,
    }
    token = jwt.encode(
        payload,
        settings.JWT_SECRET_KEY,
        algorithm=settings.JWT_ALGORITHM,
    )
    return token, expires_at, token_jti


def create_access_token(
    *,
    user_id: int,
    organization_id: int,
    role: UserRole,
) -> str:
    token, _, _ = _create_token(
        user_id=user_id,
        organization_id=organization_id,
        role=role,
        token_type="access",
        expires_delta=timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES),
    )
    return token


def create_refresh_token(
    *,
    user_id: int,
    organization_id: int,
    role: UserRole,
) -> tuple[str, datetime, str]:
    return _create_token(
        user_id=user_id,
        organization_id=organization_id,
        role=role,
        token_type="refresh",
        expires_delta=timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS),
    )


def decode_token(token: str, expected_type: str) -> dict:
    try:
        payload = jwt.decode(
            token,
            settings.JWT_SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM],
        )
    except JWTError as exc:
        raise InvalidTokenError("Invalid token") from exc

    if payload.get("type") != expected_type:
        raise InvalidTokenError("Invalid token type")

    if not payload.get("sub") or not payload.get("organization_id"):
        raise InvalidTokenError("Token is missing required claims")

    return payload
