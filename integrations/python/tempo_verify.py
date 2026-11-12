"""Tempo server-side helpers for your own backend. One file, standard library only: copy it into your project.

    from tempo_verify import verify, issue_ticket

    result = verify(os.environ["TEMPO_SECRET"], request.form.get("tempo_token", ""), "https://tempo.example.com")
    if result.success and result.user_id == account.tempo_id and result.rhythm_match:
        ...  # rhythm looks right
    else:
        ...  # your policy: ask for a second factor, or let the password alone through

Neither function raises on network trouble. When Tempo cannot be reached, answers badly, or says no, you get
success=False and a short `reason` ("unreachable", "unauthorized", "invalid_token", "rate_limited", "bad_response",
"no_token", "bad_config"), so YOU choose fail-open or fail-closed. Call verify() only after the password was checked,
compare result.user_id with the account that is logging in, and remember a token is single use and lives 60 s.

Server calls send no Origin header (Tempo refuses those), ignore proxy environment variables, do not follow redirects
(the secret is never re-sent anywhere else), and never log or return the secret.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

MAX_RESPONSE_BYTES = 65536


@dataclass(frozen=True)
class VerifyResult:
    success: bool
    user_id: str | None = None
    rhythm_match: bool = False
    risk: str | None = None  # "low" | "medium" | "high" when success
    reason: str | None = None  # why success is False


@dataclass(frozen=True)
class TicketResult:
    success: bool
    ticket: str | None = None  # hand this to the page for tempo.enroll(...)
    expires_in: int | None = None
    reason: str | None = None  # "already_enrolled" (pass reset=True), "unauthorized", "unreachable", ...


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        return None


_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())


def _post(base_url: str, path: str, payload: dict[str, Any], timeout: float) -> tuple[int, dict[str, Any] | None]:
    """(http status, JSON object or None). Status 0 means Tempo could not be reached (refused, DNS, timeout, TLS)."""
    body = json.dumps(payload).encode()
    request = urllib.request.Request(
        base_url.rstrip("/") + path, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with _OPENER.open(request, timeout=timeout) as response:
            status = response.status
            raw = response.read(MAX_RESPONSE_BYTES)
    except urllib.error.HTTPError as err:
        status = err.code
        raw = err.read(MAX_RESPONSE_BYTES)
        err.close()
    except (urllib.error.URLError, OSError, ValueError):
        return 0, None  # TimeoutError and ConnectionError are OSError
    try:
        data = json.loads(raw)
    except ValueError:
        return status, None
    if not isinstance(data, dict):
        return status, None
    return status, data


def _error_reason(status: int, data: dict[str, Any] | None) -> str:
    if status == 0:
        return "unreachable"
    if status == 401:
        return "unauthorized"
    if status == 429:
        return "rate_limited"
    if status == 409 and data is not None and isinstance(data.get("error"), dict) and data["error"].get("code") == "already_enrolled":
        return "already_enrolled"
    return "bad_response"


def _bad_base_url(base_url: str) -> bool:
    return not isinstance(base_url, str) or not base_url.startswith(("http://", "https://"))


def verify(secret: str, token: str, base_url: str, timeout: float = 3.0) -> VerifyResult:
    """Ask Tempo what the browser's token says (POST /v1/siteverify). Never raises on network or protocol errors."""
    if _bad_base_url(base_url):
        return VerifyResult(False, reason="bad_config")
    if not isinstance(token, str) or not token:
        return VerifyResult(False, reason="no_token")  # the page did not produce a token (fail-open/closed is yours)
    status, data = _post(base_url, "/v1/siteverify", {"secret": secret, "token": token}, timeout)
    if status != 200 or data is None:
        return VerifyResult(False, reason=_error_reason(status, data))
    if data.get("success") is not True:
        reason = data.get("error")
        if not isinstance(reason, str):
            reason = "invalid_token"
        return VerifyResult(False, reason=reason)
    user_id = data.get("user_id")
    risk = data.get("risk")
    if not isinstance(user_id, str) or not isinstance(data.get("rhythm_match"), bool) or not isinstance(risk, str):
        return VerifyResult(False, reason="bad_response")
    return VerifyResult(True, user_id=user_id, rhythm_match=data["rhythm_match"], risk=risk)


def issue_ticket(secret: str, user_id: str, base_url: str, reset: bool = False, timeout: float = 3.0) -> TicketResult:
    """Get an enrollment ticket for an already authenticated user (POST /v1/tickets).

    reset=True deletes the user's existing rhythm template first (re-enroll, or erase it for good: the ticket may
    simply expire unused). Call this only for the logged-in user, never with an id taken from the request.
    """
    if _bad_base_url(base_url):
        return TicketResult(False, reason="bad_config")
    status, data = _post(base_url, "/v1/tickets", {"secret": secret, "user_id": user_id, "reset": reset}, timeout)
    if status != 200 or data is None:
        return TicketResult(False, reason=_error_reason(status, data))
    ticket = data.get("ticket")
    expires_in = data.get("expires_in")
    if not isinstance(ticket, str) or not isinstance(expires_in, int):
        return TicketResult(False, reason="bad_response")
    return TicketResult(True, ticket=ticket, expires_in=expires_in)
