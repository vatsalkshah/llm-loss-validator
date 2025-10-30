"""Auth and logging middleware for the inference API."""
from __future__ import annotations

import time
from typing import Callable, Optional

from fastapi import Request, Response, HTTPException, status
from fastapi.security import APIKeyHeader
from loguru import logger
from starlette.middleware.base import BaseHTTPMiddleware

from .config import get_settings


class APIKeyValidator:
    """Simple API key guard that integrates with FastAPI dependencies."""

    def __init__(self) -> None:
        settings = get_settings()
        self.require_api_key = settings.require_api_key and settings.api_key is not None
        self.api_key = settings.api_key
        self.header = APIKeyHeader(name=settings.api_key_header, auto_error=False)

    async def __call__(self, request: Request) -> None:
        if not self.require_api_key:
            return
        token = await self.header.__call__(request)
        if not token:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="API key required")
        if token != self.api_key:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid API key")


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log inbound requests and basic timing information."""

    async def dispatch(self, request: Request, call_next: Callable[[Request], Response]) -> Response:
        start = time.time()
        response: Optional[Response] = None
        try:
            response = await call_next(request)
            return response
        finally:
            duration_ms = (time.time() - start) * 1000
            logger.info(
                "%s %s -> %s in %.2fms",
                request.method,
                request.url.path,
                getattr(response, "status_code", "?"),
                duration_ms,
            )


__all__ = ["APIKeyValidator", "RequestLoggingMiddleware"]
