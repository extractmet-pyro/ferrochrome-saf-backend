from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select

from ...core.security import current_user, issue_token, verify_password
from ...db import models as M
from ...db.session import get_db

router = APIRouter(prefix="/auth", tags=["auth"])


class LoginIn(BaseModel):
    username: str
    password: str


class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: str
    username: str


@router.post("/login", response_model=TokenOut, operation_id="auth_login")
def login(body: LoginIn, db=Depends(get_db)):
    u = db.scalar(select(M.User).where(M.User.username == body.username))
    if not u or not verify_password(body.password, u.password_hash):
        raise HTTPException(401, "invalid credentials")
    return TokenOut(access_token=issue_token(u.username, u.role, u.plant), role=u.role, username=u.username)


@router.get("/me", operation_id="auth_me")
def me(user=Depends(current_user)):
    return user
