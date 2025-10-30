"""Application configuration for the inference API."""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional

from pydantic import Field

try:  # pragma: no cover - optional dependency
    from pydantic_settings import BaseSettings
except ImportError:  # pragma: no cover - fallback for environments without pydantic-settings
    from pydantic import BaseModel as BaseSettings


class ServerSettings(BaseSettings):
    """Settings loaded from environment variables or a dotenv file."""

    host: str = Field(default_factory=lambda: os.getenv("INFERENCE_HOST", "0.0.0.0"))
    port: int = Field(default_factory=lambda: int(os.getenv("INFERENCE_PORT", "8000")))
    workers: int = Field(default_factory=lambda: int(os.getenv("INFERENCE_WORKERS", "1")))
    api_key: Optional[str] = Field(default=os.getenv("API_KEY"))
    api_key_header: str = Field(default_factory=lambda: os.getenv("API_KEY_HEADER", "X-API-Key"))
    require_api_key: bool = Field(default_factory=lambda: os.getenv("REQUIRE_API_KEY", "false").lower() == "true")
    telemetry_url: Optional[str] = Field(default=os.getenv("TELEMETRY_URL"))
    tls_cert_path: Optional[str] = Field(default=os.getenv("TLS_CERT_PATH"))
    tls_key_path: Optional[str] = Field(default=os.getenv("TLS_KEY_PATH"))
    cache_dir: Optional[str] = Field(default=os.getenv("CACHE_DIR"))
    default_model: str = Field(default_factory=lambda: os.getenv("DEFAULT_MODEL", "echo-model"))

    class Config:
        env_file = os.getenv("ENV_FILE", None)
        env_file_encoding = "utf-8"
        case_sensitive = False

    def tls_enabled(self) -> bool:
        return bool(self.tls_cert_path and self.tls_key_path)


@lru_cache(maxsize=1)
def get_settings() -> ServerSettings:
    return ServerSettings()


__all__ = ["ServerSettings", "get_settings"]
