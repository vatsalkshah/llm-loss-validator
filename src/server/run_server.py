"""CLI entrypoint to run the inference API using uvicorn."""
from __future__ import annotations

import argparse
import uvicorn

from .config import get_settings


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the inference API server")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args(argv)

    settings = get_settings()
    host = args.host or settings.host
    port = args.port or settings.port

    uvicorn.run(
        "server.main:app",
        host=host,
        port=port,
        reload=args.reload,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
