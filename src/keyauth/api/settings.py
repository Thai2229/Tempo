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
    per_user: bool = True  # per-user thresholds from the checkpoint's params (global threshold if it has none)
    app_id: str = "default"  # single-tenant demo: every request uses this app_id
    debug_scores: bool = False  # include score / threshold in /verify responses
    rate_limit: int = 30  # max requests per window, per client IP and per user_id
    rate_window_s: float = 60.0
    admin_token: str | None = None  # DELETE /user needs "Authorization: Bearer <token>"; unset = refused
    serve_ui: bool = True  # serve the demo page at / and its assets under /static
    landing: bool = False  # with serve_ui: the landing page at /, the demo page moves to /demo
    enable_v1: bool = False  # mount the multi-site service under /v1 (needs token_key)
    token_key: str | None = None  # HMAC key signing /v1 tokens; 32+ characters, no default on purpose
    server_rate_limit: int = 600  # /v1/siteverify and /v1/tickets: requests per window, per client IP and per site
    server_rate_window_s: float = 60.0

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "Settings":
        if env is None:
            env = os.environ
        limit, window = parse_rate_limit(env.get("KEYAUTH_RATE_LIMIT", "30/60s"))
        admin_token = env.get("KEYAUTH_ADMIN_TOKEN") or None
        server_limit, server_window = parse_rate_limit(env.get("KEYAUTH_SERVER_RATE_LIMIT", "600/60s"))
        return cls(
            checkpoint=env.get("KEYAUTH_CHECKPOINT", "checkpoints/model.pt"),
            db_path=env.get("KEYAUTH_DB", "keyauth.db"),
            per_user=env.get("KEYAUTH_PER_USER", "1").strip().lower() in _TRUE,
            app_id=env.get("KEYAUTH_APP_ID", "default"),
            debug_scores=env.get("KEYAUTH_DEBUG_SCORES", "0").strip().lower() in _TRUE,
            rate_limit=limit,
            rate_window_s=window,
            admin_token=admin_token,
            serve_ui=env.get("KEYAUTH_SERVE_UI", "1").strip().lower() in _TRUE,
            landing=env.get("KEYAUTH_LANDING", "0").strip().lower() in _TRUE,
            enable_v1=env.get("KEYAUTH_ENABLE_V1", "0").strip().lower() in _TRUE,
            token_key=env.get("KEYAUTH_TOKEN_KEY") or None,
            server_rate_limit=server_limit,
            server_rate_window_s=server_window,
        )
