"""
Configuration for the inference API server.

This module manages server configuration including API authentication,
TLS settings, logging controls, and model loading parameters.
"""

import os
from typing import Optional
from pathlib import Path
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()


class ServerConfig(BaseSettings):
    """Configuration settings for the inference server."""
    
    # Server settings
    host: str = os.getenv("SERVER_HOST", "0.0.0.0")
    port: int = int(os.getenv("SERVER_PORT", "8000"))
    workers: int = int(os.getenv("SERVER_WORKERS", "1"))
    
    # API Authentication
    api_key: Optional[str] = os.getenv("API_KEY")
    api_key_header: str = os.getenv("API_KEY_HEADER", "X-API-Key")
    require_api_key: bool = os.getenv("REQUIRE_API_KEY", "false").lower() == "true"
    
    # TLS/SSL Configuration
    tls_enabled: bool = os.getenv("TLS_ENABLED", "false").lower() == "true"
    tls_cert_path: Optional[str] = os.getenv("TLS_CERT_PATH")
    tls_key_path: Optional[str] = os.getenv("TLS_KEY_PATH")
    
    # Logging and security
    log_requests: bool = os.getenv("LOG_REQUESTS", "true").lower() == "true"
    log_responses: bool = os.getenv("LOG_RESPONSES", "false").lower() == "true"
    log_payloads: bool = os.getenv("LOG_PAYLOADS", "false").lower() == "true"
    
    # Model settings
    hf_token: Optional[str] = os.getenv("HF_TOKEN")
    default_max_tokens: int = int(os.getenv("DEFAULT_MAX_TOKENS", "512"))
    default_temperature: float = float(os.getenv("DEFAULT_TEMPERATURE", "0.7"))
    max_context_length: int = int(os.getenv("MAX_CONTEXT_LENGTH", "4096"))
    
    # Cache settings (delegated to ModelCacheManager)
    cache_dir: Optional[str] = os.getenv("CACHE_DIR")
    cache_enabled: bool = os.getenv("CACHE_ENABLED", "true").lower() == "true"
    
    # Generation limits
    max_gen_tokens_limit: int = int(os.getenv("MAX_GEN_TOKENS_LIMIT", "2048"))
    
    # Server mode
    reject_non_cached_models: bool = os.getenv("REJECT_NON_CACHED_MODELS", "true").lower() == "true"
    
    class Config:
        env_file = ".env"
        case_sensitive = False
    
    def validate_tls_config(self) -> bool:
        """Validate TLS configuration if TLS is enabled."""
        if not self.tls_enabled:
            return True
        
        if not self.tls_cert_path or not self.tls_key_path:
            return False
        
        cert_exists = Path(self.tls_cert_path).exists()
        key_exists = Path(self.tls_key_path).exists()
        
        return cert_exists and key_exists


# Global config instance
config = ServerConfig()
