"""Admin session tokens (HMAC). Patient portal does not use this."""

from __future__ import annotations

import hashlib
import hmac
import time

from fastapi import Header, HTTPException

from .config import get_settings

TOKEN_TTL_SEC = 12 * 3600


def issue_admin_token() -> str:
    settings = get_settings()
    secret = (settings.jwt_secret or settings.admin_password or "dev").encode()
    issued = str(int(time.time()))
    sig = hmac.new(secret, issued.encode(), hashlib.sha256).hexdigest()
    return f"{issued}.{sig}"


def verify_admin_token(token: str) -> bool:
    settings = get_settings()
    secret = (settings.jwt_secret or settings.admin_password or "dev").encode()
    try:
        issued_s, sig = token.split(".", 1)
        issued = int(issued_s)
    except ValueError:
        return False
    if time.time() - issued > TOKEN_TTL_SEC:
        return False
    expect = hmac.new(secret, issued_s.encode(), hashlib.sha256).hexdigest()
    return hmac.compare_digest(expect, sig)


def require_admin(authorization: str | None = Header(default=None)) -> None:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Admin login required.")
    token = authorization.split(" ", 1)[1].strip()
    if not verify_admin_token(token):
        raise HTTPException(status_code=401, detail="Invalid or expired admin token.")
