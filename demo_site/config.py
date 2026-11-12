"""Settings of the demo site, from environment variables only. The site secret lives here and nowhere else."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from typing import Mapping
from urllib.parse import urlsplit

_TRUE = {"1", "true", "yes", "on"}


def origin_of(url: str) -> str:
    """'http://127.0.0.1:8765/some/path' -> 'http://127.0.0.1:8765' (what a CSP source and a CORS origin look like)."""
    parts = urlsplit(url.strip())
    if parts.scheme not in {"http", "https"} or not parts.netloc:
        raise ValueError(f"TEMPO_URL must look like https://tempo.example, got {url!r}")
    return f"{parts.scheme}://{parts.netloc}"


@dataclass(frozen=True)
class SiteConfig:
    tempo_url: str
    site_secret: str = field(repr=False)  # never rendered, never logged
    site_key: str
    db_path: str = "demo_site.db"
    host: str = "127.0.0.1"
    port: int = 9000
    fail_mode: str = "open"  # what an empty token or an unreachable tempo means: "open" or "closed"
    secure_cookie: bool = False
    pepper: bytes = field(default=b"", repr=False)

    @staticmethod
    def from_env(env: Mapping[str, str] | None = None) -> "SiteConfig":
        if env is None:
            env = os.environ
        missing = [name for name in ("TEMPO_URL", "TEMPO_SITE_SECRET", "TEMPO_SITE_KEY") if not env.get(name)]
        if missing:
            raise ValueError("missing environment variables: " + ", ".join(missing) + " (see demo_site/README.md)")
        fail_mode = env.get("DEMO_SITE_FAIL_MODE", "open").strip().lower()
        if fail_mode not in {"open", "closed"}:
            raise ValueError("DEMO_SITE_FAIL_MODE must be 'open' or 'closed'")
        return SiteConfig(
            tempo_url=origin_of(env["TEMPO_URL"]),
            site_secret=env["TEMPO_SITE_SECRET"],
            site_key=env["TEMPO_SITE_KEY"],
            db_path=env.get("DEMO_SITE_DB", "demo_site.db"),
            host=env.get("DEMO_SITE_HOST", "127.0.0.1"),
            port=int(env.get("DEMO_SITE_PORT", "9000")),
            fail_mode=fail_mode,
            secure_cookie=env.get("DEMO_SITE_SECURE_COOKIE", "0").strip().lower() in _TRUE,
            pepper=derive_pepper(env.get("DEMO_SITE_PEPPER") or env["TEMPO_SITE_SECRET"]),
        )


def derive_pepper(seed: str) -> bytes:
    """Key for the stand-in ids shown for unknown usernames (domain separated, so it is not the secret itself)."""
    return hashlib.sha256(b"demo-site-identify-v1." + seed.encode()).digest()
