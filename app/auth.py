"""Single-user login: bcrypt hash from .env, signed session cookie."""
import hmac

import bcrypt
from fastapi import Request

from . import config


def check_credentials(username: str, password: str) -> bool:
    if not config.APP_PASSWORD_HASH:
        return False
    user_ok = hmac.compare_digest(username, config.APP_USERNAME)
    pw_ok = bcrypt.checkpw(password.encode(), config.APP_PASSWORD_HASH.encode())
    return user_ok and pw_ok


def current_user(request: Request) -> str | None:
    return request.session.get("user")
