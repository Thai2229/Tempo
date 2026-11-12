"""`python -m demo_site` (run from the repository root). Configuration: environment variables only, see README.md."""

from __future__ import annotations

import logging

import uvicorn

from demo_site.app import create_app
from demo_site.config import SiteConfig


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    config = SiteConfig.from_env()
    print(f"demo site: http://{config.host}:{config.port}/  (tempo at {config.tempo_url})", flush=True)
    uvicorn.run(create_app(config), host=config.host, port=config.port)


if __name__ == "__main__":
    main()
