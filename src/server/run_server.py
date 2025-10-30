#!/usr/bin/env python
"""
Convenience script to run the inference API server.

Usage:
    python run_server.py
    python run_server.py --host 0.0.0.0 --port 9000
"""

import argparse
import uvicorn

from server.config import config


def main():
    parser = argparse.ArgumentParser(description="Run the inference API server")
    parser.add_argument("--host", default=config.host, help="Host to bind to")
    parser.add_argument("--port", type=int, default=config.port, help="Port to bind to")
    parser.add_argument("--workers", type=int, default=config.workers, help="Number of workers")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    
    args = parser.parse_args()
    
    uvicorn_config = {
        "app": "server.main:app",
        "host": args.host,
        "port": args.port,
        "workers": args.workers if not args.reload else 1,
        "log_level": "info",
        "reload": args.reload,
    }
    
    # Add TLS if enabled
    if config.tls_enabled:
        if config.validate_tls_config():
            uvicorn_config["ssl_certfile"] = config.tls_cert_path
            uvicorn_config["ssl_keyfile"] = config.tls_key_path
            print("TLS enabled")
        else:
            print("TLS configuration invalid, starting without TLS")
    
    uvicorn.run(**uvicorn_config)


if __name__ == "__main__":
    main()
