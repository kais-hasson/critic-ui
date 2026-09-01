from datetime import datetime, timedelta, timezone
from typing import Optional

import jwt
from pwdlib import PasswordHash


# ============================================================
# Password Hashing
# ============================================================

password_hash = PasswordHash.recommended()


def hash_password(password: str) -> str:
    return password_hash.hash(password)


def verify_password(
    plain_password: str,
    hashed_password: str,
) -> bool:
    return password_hash.verify(
        plain_password,
        hashed_password,
    )


# ============================================================
# JWT Configuration
# ============================================================

SECRET_KEY = "CHANGE_THIS_LATER"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60


# ============================================================
# Create Access Token
# ============================================================

def create_access_token(
    data: dict,
    expires_delta: Optional[timedelta] = None,
) -> str:

    to_encode = data.copy()

    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(
            minutes=ACCESS_TOKEN_EXPIRE_MINUTES
        )

    to_encode.update({"exp": expire})

    encoded_jwt = jwt.encode(
        to_encode,
        SECRET_KEY,
        algorithm=ALGORITHM,
    )

    return encoded_jwt


# ============================================================
# Decode Access Token
# ============================================================

def decode_access_token(token: str) -> dict:
    return jwt.decode(
        token,
        SECRET_KEY,
        algorithms=[ALGORITHM],
    )