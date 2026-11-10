"""Runtime settings for the HTTP API, read from environment variables (see .env.example)."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Mapping

_RATE_LIMIT = re.compile(r"^\s*(\d+)\s*/\s*(\d+(?:\.\d+)?)\s*s?\s*$")
_TRUE = frozenset({"1", "true", "yes", "on"})


def parse_rate_limit(text: str) -> tuple[int, float]:
    """'30/60s' (or '30/60') -> (30 requests, 60.0 seconds)."""
    match = _RATE_LIMIT.match(text)
    if match is None or int(match.group(1)) < 1 or float(match.group(2)) <= 0:
        raise ValueError(f"KEYAUTH_RATE_LIMIT must look like '30/60s', got {text!r}")
    return int(match.group(1)), float(match.group(2))


@dataclass
class Settings:
    checkpoint: str = "checkpoints/model.pt"
    db_path: str = "keyauth.db"
    per_user: bool = False  # per-user thresholds from the checkpoint's params (opt-in)
    app_id: str = "default"  # single-tenant demo: every request uses this app_id
    debug_scores: bool = False  # include score / threshold in /verify responses
    rate_limit: int = 30  # max requests per window, per client IP and per user_id
    rate_window_s: float = 60.0
    admin_token: str | None = None  # if set, DELETE /user needs "Authorization: Bearer <token>"
    serve_ui: bool = True  # serve the demo page at / and its assets under /static

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "Settings":
        if env is None:
            env = os.environ
        limit, window = parse_rate_limit(env.get("KEYAUTH_RATE_LIMIT", "30/60s"))
        admin_token = env.get("KEYAUTH_ADMIN_TOKEN") or None
        return cls(
            checkpoint=env.get("KEYAUTH_CHECKPOINT", "checkpoints/model.pt"),
            db_path=env.get("KEYAUTH_DB", "keyauth.db"),
            per_user=env.get("KEYAUTH_PER_USER", "0").strip().lower() in _TRUE,
            app_id=env.get("KEYAUTH_APP_ID", "default"),
            debug_scores=env.get("KEYAUTH_DEBUG_SCORES", "0").strip().lower() in _TRUE,
            rate_limit=limit,
            rate_window_s=window,
            admin_token=admin_token,
            serve_ui=env.get("KEYAUTH_SERVE_UI", "1").strip().lower() in _TRUE,
        )
