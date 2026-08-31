"""Grant or revoke the ``admin`` custom claim on a Firebase account.

``require_admin`` in auth.py authorizes on ``claims.get("admin")``, but nothing
in the running application ever writes that claim: Firebase custom claims can
only be set from a trusted server with the Admin SDK, never from the browser.
Without this script ``POST /index/folder`` is unreachable in production.

``LEGACY_ADMIN_UID`` is a different setting and is easy to confuse with this
one. It decides which UID inherits the pre-tenancy SQLite and Chroma rows; it
does not confer the administrator role.

READ ONLY BY DEFAULT. The claim lets an account index arbitrary server-local
folders under INDEX_FOLDER_ROOTS, so --grant and --revoke are opt-in.
"""

from __future__ import annotations

import argparse
import sys

from rich.console import Console

console = Console()


def _resolve_user(firebase_auth, identifier: str):
    """Accept either a Firebase UID or the email the account signed in with."""
    if "@" in identifier:
        return firebase_auth.get_user_by_email(identifier)
    return firebase_auth.get_user(identifier)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inspect or change the admin custom claim on a Firebase account.",
    )
    parser.add_argument(
        "identifier",
        help="Firebase UID, or the Google email address the account signed in with.",
    )
    action = parser.add_mutually_exclusive_group()
    action.add_argument(
        "--grant", action="store_true", help="Add the admin claim to this account."
    )
    action.add_argument(
        "--revoke", action="store_true", help="Remove the admin claim from this account."
    )
    args = parser.parse_args()

    from auth import _firebase_auth

    try:
        firebase_auth = _firebase_auth()
    except RuntimeError as exc:
        console.print(f"[red]❌ {exc}[/red]")
        return 1

    try:
        user = _resolve_user(firebase_auth, args.identifier)
    except Exception as exc:
        console.print(f"[red]❌ No Firebase account matched {args.identifier!r}.[/red]")
        console.print(f"   {type(exc).__name__}: {exc}")
        return 1

    claims = dict(user.custom_claims or {})
    currently_admin = claims.get("admin") is True
    console.print(f"👤 {user.uid}  {user.email or '(no email)'}")
    console.print(f"   admin claim: [bold]{currently_admin}[/bold]")

    if not (args.grant or args.revoke):
        console.print("   Pass --grant or --revoke to change it.")
        return 0

    if args.grant:
        claims["admin"] = True
    else:
        claims.pop("admin", None)

    # Passing None clears the claim document rather than storing an empty one.
    firebase_auth.set_custom_user_claims(user.uid, claims or None)
    console.print(f"   ✅ admin claim is now [bold]{claims.get('admin') is True}[/bold]")

    if args.revoke:
        # A claim lives inside already-issued tokens and session cookies, so
        # revoking it only takes effect once those are refreshed. Revoking the
        # refresh tokens forces that, and the API only notices when
        # AUTH_CHECK_REVOKED is enabled.
        firebase_auth.revoke_refresh_tokens(user.uid)
        console.print("   ↩️  Refresh tokens revoked; the account must sign in again.")
        console.print("   ⚠️  Set AUTH_CHECK_REVOKED=1 so the API enforces this.")
    else:
        console.print("   ℹ️  Sign out and in again to pick the claim up in a session.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
