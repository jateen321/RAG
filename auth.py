"""Firebase-backed browser sessions and FastAPI authorization dependencies."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import time

from fastapi import Depends, HTTPException, Request, status

from config import (
    AUTH_CHECK_REVOKED,
    AUTH_RECENT_SIGN_IN_MAX_AGE_S,
    FIREBASE_PROJECT_ID,
    SESSION_COOKIE_NAME,
)


@dataclass(frozen=True)
class AuthenticatedUser:
    """Identity and application role extracted from a verified Firebase session."""

    uid: str
    email: str | None = None
    is_admin: bool = False


@lru_cache(maxsize=1)
def _firebase_auth():
    """Initialize Firebase Admin lazily so CLI tools do not require auth setup."""
    if not FIREBASE_PROJECT_ID:
        raise RuntimeError("FIREBASE_PROJECT_ID is required for API authentication.")

    import firebase_admin
    from firebase_admin import auth, credentials

    try:
        firebase_admin.get_app()
    except ValueError:
        # Application Default Credentials are the production path on Google
        # Cloud. Locally, GOOGLE_APPLICATION_CREDENTIALS may point at a service
        # account without putting its secret in this repository.
        firebase_admin.initialize_app(
            credentials.ApplicationDefault(),
            {"projectId": FIREBASE_PROJECT_ID},
        )
    return auth


def create_session_cookie(id_token: str, expires_in_seconds: int) -> str:
    firebase_auth = _firebase_auth()
    try:
        claims = firebase_auth.verify_id_token(
            id_token,
            check_revoked=AUTH_CHECK_REVOKED,
        )
        auth_time = int(claims.get("auth_time", 0))
        provider = (claims.get("firebase") or {}).get("sign_in_provider")
        if provider != "google.com":
            raise ValueError("Only Google sign-in is supported.")
        if auth_time <= 0 or time.time() - auth_time > AUTH_RECENT_SIGN_IN_MAX_AGE_S:
            raise ValueError("Sign in with Google again before creating a session.")
        return firebase_auth.create_session_cookie(
            id_token,
            expires_in=expires_in_seconds,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Google sign-in could not be verified.",
        ) from exc


def verify_session_cookie(session_cookie: str) -> AuthenticatedUser:
    try:
        claims = _firebase_auth().verify_session_cookie(
            session_cookie,
            check_revoked=AUTH_CHECK_REVOKED,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Your session is missing or has expired. Please sign in again.",
        ) from exc

    uid = str(claims.get("uid") or claims.get("sub") or "").strip()
    if not uid:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="The verified session does not identify a user.",
        )
    return AuthenticatedUser(
        uid=uid,
        email=str(claims["email"]) if claims.get("email") else None,
        is_admin=claims.get("admin") is True,
    )


def get_current_user(request: Request) -> AuthenticatedUser:
    cookie = request.cookies.get(SESSION_COOKIE_NAME, "")
    if not cookie:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Sign in to continue.",
        )
    return verify_session_cookie(cookie)


def require_admin(
    user: AuthenticatedUser = Depends(get_current_user),
) -> AuthenticatedUser:
    if not user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Administrator access is required.",
        )
    return user
