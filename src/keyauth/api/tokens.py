"""Compact signed tokens for the multi-site service (see docs/SERVICE.md).

Format:  base64url(payload_json) "." base64url(HMAC-SHA256(key, "keyauth-token-v1." + base64url(payload_json)))
There is no algorithm field: the algorithm is fixed, so a client cannot downgrade it. The MAC covers the
exact payload text that is later parsed, and it is compared in constant time before anything is parsed.

Payload: v=1, sid (site id), uid (opaque user id), m (rhythm match, bool), r (risk: low|medium|high),
iat / exp (seconds since the epoch), jti (random id, consumed once by /v1/siteverify).
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
from dataclasses import dataclass

TOKEN_TTL_S = 60.0
MAX_TOKEN_CHARS = 2048
CLOCK_SKEW_S = 5.0
RISKS = frozenset({"low", "medium", "high"})
_DOMAIN = b"keyauth-token-v1."


@dataclass
class TokenClaims:
    site_id: str
    user_id: str
    rhythm_match: bool
    risk: str
    issued_at: float
    expires_at: float
    jti: str


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _unb64(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def _mac(key: bytes, payload_b64: str) -> str:
    return _b64(hmac.new(key, _DOMAIN + payload_b64.encode(), hashlib.sha256).digest())


def new_jti() -> str:
    return secrets.token_hex(16)


def sign_token(key: bytes, claims: TokenClaims) -> str:
    payload = {
        "v": 1,
        "sid": claims.site_id,
        "uid": claims.user_id,
        "m": claims.rhythm_match,
        "r": claims.risk,
        "iat": claims.issued_at,
        "exp": claims.expires_at,
        "jti": claims.jti,
    }
    payload_b64 = _b64(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode())
    return payload_b64 + "." + _mac(key, payload_b64)


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def verify_token(key: bytes, token: str, now: float) -> TokenClaims | None:
    """Claims of a genuine, unexpired token, else None (the caller must not learn why)."""
    if not isinstance(token, str) or len(token) > MAX_TOKEN_CHARS or token.count(".") != 1:
        return None
    payload_b64, given = token.split(".")
    if not hmac.compare_digest(_mac(key, payload_b64).encode(), given.encode("ascii", "replace")):
        return None
    try:
        data = json.loads(_unb64(payload_b64))
    except ValueError:
        return None
    if not isinstance(data, dict) or data.get("v") != 1:
        return None
    sid, uid, match, risk = data.get("sid"), data.get("uid"), data.get("m"), data.get("r")
    iat, exp, jti = data.get("iat"), data.get("exp"), data.get("jti")
    if (
        not isinstance(sid, str)
        or not isinstance(uid, str)
        or not isinstance(match, bool)
        or risk not in RISKS
        or not _is_number(iat)
        or not _is_number(exp)
        or not isinstance(jti, str)
        or len(jti) < 16
        or exp <= now
        or exp - iat > TOKEN_TTL_S
        or iat > now + CLOCK_SKEW_S
    ):
        return None
    return TokenClaims(sid, uid, match, risk, float(iat), float(exp), jti)
