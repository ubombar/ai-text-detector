"""Create .env with a username, bcrypt password hash and session secret.

    uv run python -m app.setup_env
"""
import getpass
import secrets

import bcrypt

from .config import ROOT

user = input("Username [admin]: ").strip() or "admin"
pw = getpass.getpass("Password: ")
if not pw or pw != getpass.getpass("Repeat password: "):
    raise SystemExit("Passwords empty or do not match.")
hashed = bcrypt.hashpw(pw.encode(), bcrypt.gensalt()).decode()
(ROOT / ".env").write_text(
    f"APP_USERNAME={user}\nAPP_PASSWORD_HASH='{hashed}'\nSECRET_KEY={secrets.token_hex(32)}\n"
)
print("Wrote .env")
