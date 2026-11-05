"""JWT (HS256, server-side verification) + PBKDF2 password hashing + RBAC.

Roles (ascending privilege): viewer < operator < metallurgist < engineer < administrator.
For OIDC deployments, replace `issue_token`/`login` with the identity provider and keep
`require_role` (maps IdP group claims to these roles).
"""
from __future__ import annotations

import hashlib
import hmac
import os
import time

import jwt
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from .config import settings

ROLES = ["viewer", "operator", "metallurgist", "engineer", "administrator"]
bearer = HTTPBearer(auto_error=False)


def hash_password(pw: str, salt: bytes | None = None) -> str:
    salt = salt or os.urandom(16)
    dk = hashlib.pbkdf2_hmac("sha256", pw.encode(), salt, 200_000)
    return salt.hex() + ":" + dk.hex()


def verify_password(pw: str, stored: str) -> bool:
    salt_hex, dk_hex = stored.split(":")
    dk = hashlib.pbkdf2_hmac("sha256", pw.encode(), bytes.fromhex(salt_hex), 200_000)
    return hmac.compare_digest(dk.hex(), dk_hex)


def issue_token(username: str, role: str, plant: str = "industry-x") -> str:
    now = int(time.time())
    return jwt.encode({"sub": username, "role": role, "plant": plant, "iat": now, "exp": now + settings.jwt_ttl_min * 60,
                       "iss": "fecr-twin"}, settings.secret_key, algorithm="HS256")


def decode(token: str) -> dict:
    try:
        return jwt.decode(token, settings.secret_key, algorithms=["HS256"], issuer="fecr-twin")
    except jwt.PyJWTError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=f"invalid token: {exc}") from exc


def current_user(request: Request, cred: HTTPAuthorizationCredentials | None = Depends(bearer)) -> dict:
    if cred is None:
        raise HTTPException(status_code=401, detail="authentication required")
    u = decode(cred.credentials)
    request.state.user = u
    return u


def require_role(min_role: str):
    def dep(user: dict = Depends(current_user)) -> dict:
        if ROLES.index(user.get("role", "viewer")) < ROLES.index(min_role):
            raise HTTPException(status_code=403, detail=f"role '{min_role}' or higher required")
        return user
    return dep
