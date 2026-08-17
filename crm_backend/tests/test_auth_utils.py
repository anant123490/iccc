from app.auth.jwt import create_access_token, decode_token
from app.auth.password import hash_password, verify_password
from app.auth.token_hash import hash_token
from app.models.enums import UserRole


def test_password_hashing_round_trip() -> None:
    hashed_password = hash_password("correct-password")

    assert hashed_password != "correct-password"
    assert verify_password("correct-password", hashed_password)
    assert not verify_password("wrong-password", hashed_password)


def test_access_token_contains_tenant_claim() -> None:
    token = create_access_token(
        user_id=7,
        organization_id=3,
        role=UserRole.OWNER,
    )

    payload = decode_token(token, expected_type="access")

    assert payload["sub"] == "7"
    assert payload["organization_id"] == 3
    assert payload["role"] == "owner"


def test_refresh_token_hash_is_stable_and_not_plain_text() -> None:
    token = "sample-refresh-token"

    assert hash_token(token) == hash_token(token)
    assert hash_token(token) != token
