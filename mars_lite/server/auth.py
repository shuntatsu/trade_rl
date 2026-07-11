"""Machine-to-machine bearer authentication for the serving plane."""

from __future__ import annotations

import secrets
from collections.abc import Callable, Coroutine
from typing import Any

from fastapi import Header, HTTPException, status


def bearer_dependency(
    expected_token: str,
) -> Callable[[str | None], Coroutine[Any, Any, None]]:
    if not expected_token:
        raise ValueError("serving bearer token must be non-empty")

    async def require_bearer_token(
        authorization: str | None = Header(default=None),
    ) -> None:
        if authorization is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="missing bearer token",
                headers={"WWW-Authenticate": "Bearer"},
            )
        scheme, separator, supplied = authorization.partition(" ")
        if not separator or scheme.lower() != "bearer" or not supplied:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="malformed bearer token",
                headers={"WWW-Authenticate": "Bearer"},
            )
        if not secrets.compare_digest(supplied, expected_token):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="invalid bearer token",
            )

    return require_bearer_token
