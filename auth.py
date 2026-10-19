"""Single-user identity and FastAPI authorization dependencies."""

from __future__ import annotations

from dataclasses import dataclass
from fastapi import Depends, HTTPException, status

from config import SINGLE_USER_ID



@dataclass(frozen=True)
class AuthenticatedUser:
    """The one configured local identity used by this single-user deployment."""

    uid: str
    email: str | None = None
    is_admin: bool = False


_SINGLE_USER = AuthenticatedUser(uid=SINGLE_USER_ID, email=None, is_admin=True)


def get_current_user() -> AuthenticatedUser:
    """Return the sole local identity; deployment networking restricts access."""
    return _SINGLE_USER


def get_optional_user() -> AuthenticatedUser:
    """Return the sole local identity for every request."""
    return _SINGLE_USER


def require_admin(
    user: AuthenticatedUser = Depends(get_current_user),
) -> AuthenticatedUser:
    if not user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Administrator access is required.",
        )
    return user
