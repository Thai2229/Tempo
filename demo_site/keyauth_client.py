"""Server-to-server calls to keyauth (/v1/tickets, /v1/siteverify). The site secret is only ever sent from here.

No Origin header is sent (keyauth refuses server calls that carry one) and proxy environment variables are ignored.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from typing import Any, Protocol

logger = logging.getLogger("demo_site")


class KeyauthLike(Protocol):
    def create_ticket(self, user_id: str, reset: bool) -> tuple[int, dict[str, Any] | None]: ...

    def siteverify(self, token: str) -> dict[str, Any] | None: ...


class KeyauthClient:
    def __init__(self, base_url: str, secret: str, timeout_s: float = 3.0) -> None:
        self.base_url = base_url.rstrip("/")
        self._secret = secret
        self.timeout_s = timeout_s
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def _post(self, path: str, payload: dict[str, Any]) -> tuple[int, dict[str, Any] | None]:
        """(http status, json body or None). Status 0 means keyauth could not be reached."""
        body = json.dumps({"secret": self._secret, **payload}).encode()
        request = urllib.request.Request(self.base_url + path, data=body, headers={"Content-Type": "application/json"}, method="POST")
        try:
            with self._opener.open(request, timeout=self.timeout_s) as response:
                status = response.status
                raw = response.read(65536)
        except urllib.error.HTTPError as err:
            status = err.code
            raw = err.read(65536)
        except (urllib.error.URLError, TimeoutError, OSError):
            return 0, None
        try:
            data = json.loads(raw)
        except ValueError:
            return status, None
        if not isinstance(data, dict):
            return status, None
        return status, data

    def create_ticket(self, user_id: str, reset: bool) -> tuple[int, dict[str, Any] | None]:
        return self._post("/v1/tickets", {"user_id": user_id, "reset": reset})

    def siteverify(self, token: str) -> dict[str, Any] | None:
        """keyauth's answer, or None when it is unreachable or answers anything but a normal 200 (misconfiguration included)."""
        status, data = self._post("/v1/siteverify", {"token": token})
        if status != 200 or data is None:
            logger.error("siteverify unavailable http_status=%s", status)
            return None
        return data
