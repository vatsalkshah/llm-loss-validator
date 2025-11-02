from __future__ import annotations

import uvicorn

from src.api.server import create_app
from src.config import get_config


def main() -> None:
    cfg = get_config()
    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=cfg.port, log_level="info")


if __name__ == "__main__":
    main()
