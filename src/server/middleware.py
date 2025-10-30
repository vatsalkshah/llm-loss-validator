"""
Middleware for the inference API server.

This module provides authentication, request/response logging,
and other cross-cutting concerns.
"""

from typing import Callable, Optional
import time

from fastapi import Request, Response, HTTPException, status
from fastapi.security import APIKeyHeader
from starlette.middleware.base import BaseHTTPMiddleware
from loguru import logger

from server.config import config


# API Key authentication
api_key_header = APIKeyHeader(name=config.api_key_header, auto_error=False)


async def verify_api_key(api_key: Optional[str] = None) -> bool:
    """
    Verify API key if authentication is enabled.
    
    Args:
        api_key: API key from request header
    
    Returns:
        True if valid or auth disabled, raises HTTPException otherwise
    """
    if not config.require_api_key:
        return True
    
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required",
        )
    
    if api_key != config.api_key:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key",
        )
    
    return True


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware for logging requests and responses.
    
    Logs request method, path, duration, and optionally payloads
    based on configuration.
    """
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and log details."""
        start_time = time.time()
        
        # Log incoming request
        if config.log_requests:
            log_msg = f"→ {request.method} {request.url.path}"
            
            if config.log_payloads:
                # For POST/PUT requests, try to read and log body
                if request.method in ["POST", "PUT", "PATCH"]:
                    try:
                        body = await request.body()
                        # Store body for later use
                        request._body = body
                        log_msg += f" | Body: {body.decode('utf-8')[:500]}"  # Limit length
                    except Exception as e:
                        logger.debug(f"Could not read request body: {e}")
            
            logger.info(log_msg)
        
        # Process request
        try:
            response = await call_next(request)
        except Exception as e:
            logger.error(f"Error processing request: {e}")
            raise
        
        # Log response
        if config.log_responses:
            duration_ms = (time.time() - start_time) * 1000
            log_msg = f"← {request.method} {request.url.path} | Status: {response.status_code} | Duration: {duration_ms:.2f}ms"
            
            if config.log_payloads and hasattr(response, "body"):
                try:
                    # Note: This is complex with streaming responses
                    log_msg += f" | Response preview available"
                except Exception as e:
                    logger.debug(f"Could not log response body: {e}")
            
            logger.info(log_msg)
        
        return response
