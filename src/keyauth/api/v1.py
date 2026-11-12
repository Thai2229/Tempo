"""Multi-site (Turnstile-style) service mounted under /v1. Flow, token format and threat model: docs/SERVICE.md.

Browser endpoints (CORS limited to the site's allowed origins; they never say "accepted"):
  POST /v1/enroll    {sitekey, user_id, ticket, events | timings}   -> {status, samples, required}
  POST /v1/verify    {sitekey, user_id, events | timings}           -> {token}
  OPTIONS on both    preflight
Server endpoints (secret in the body; no CORS; requests carrying an Origin header are refused):
  POST /v1/tickets     {secret, user_id, reset?}  -> {ticket, expires_in}
  POST /v1/siteverify  {secret, token}            -> {success, user_id, rhythm_match, risk} | {success: false, error}

Errors keep the shape {"error": {"code", "message"}}. 403 forbidden: unknown/disabled site key, missing or
disallowed Origin (all the same answer). 403 invalid_ticket. 401 unauthorized: bad secret. /v1/verify answers 200
{token} for a match, a mismatch, an unknown user, a replay, a length mismatch AND a locked user; only the signed
token differs. Lockout therefore is NOT a 423 here (it would tell the browser which ids exist and are under attack);
the customer sees rhythm_match=false, risk="high". Input that is malformed on its own (422) and rate limits (429)
do not depend on the user's secret state and are returned as usual.
"""

from __future__ import annotations

import hashlib
import logging
import secrets
import time
from typing import Annotated, Callable

from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import Response
from pydantic import BaseModel, Field

from keyauth.api.app import ApiError, AttemptIn, to_timings, user_tag
from keyauth.api.ratelimit import SlidingWindowLimiter
from keyauth.api.settings import Settings
from keyauth.api.tokens import TOKEN_TTL_S, TokenClaims, new_jti, sign_token, verify_token
from keyauth.auth import KeyAuthEngine, SiteRecord, TemplateStore
from keyauth.sites import authenticate_secret, secret_site_id

logger = logging.getLogger("keyauth.api")

MIN_TOKEN_KEY_CHARS = 32
TICKET_TTL_S = 600.0
V1_USER_ID_PATTERN = r"^[A-Za-z0-9._-]{1,64}$"  # opaque ids; no '@' on purpose (do not use emails)

V1UserId = Annotated[str, Field(min_length=1, max_length=64, pattern=V1_USER_ID_PATTERN)]


class V1AttemptIn(AttemptIn):
    sitekey: str = Field(min_length=1, max_length=64)
    user_id: V1UserId


class V1EnrollIn(V1AttemptIn):
    ticket: str = Field(min_length=1, max_length=128)


class TicketIn(BaseModel):
    secret: str = Field(min_length=1, max_length=256)
    user_id: V1UserId
    reset: bool = False  # delete an already enrolled user's template so they can enroll again


class SiteverifyIn(BaseModel):
    secret: str = Field(min_length=1, max_length=256)
    token: str = Field(min_length=1, max_length=2048)


def check_settings(settings: Settings) -> None:
    """Refuse to start the service without a signing key; there is deliberately no default."""
    if settings.token_key is None or len(settings.token_key) < MIN_TOKEN_KEY_CHARS:
        raise ValueError(
            f"KEYAUTH_ENABLE_V1 needs KEYAUTH_TOKEN_KEY with at least {MIN_TOKEN_KEY_CHARS} characters "
            "(python -c \"import secrets; print(secrets.token_urlsafe(48))\")"
        )


def ticket_hash(ticket: str) -> str:
    return hashlib.sha256(ticket.encode()).hexdigest()


def install(
    app: FastAPI,
    settings: Settings,
    store: TemplateStore,
    require_engine: Callable[[], KeyAuthEngine],
    get_engine: Callable[[], KeyAuthEngine | None],
    limiter: SlidingWindowLimiter,
) -> None:
    check_settings(settings)
    key = settings.token_key.encode()
    server_limiter = SlidingWindowLimiter(settings.server_rate_limit, settings.server_rate_window_s)
    app.state.server_limiter = server_limiter  # tests swap .clock
    router = APIRouter(prefix="/v1")

    def now() -> float:
        eng = get_engine()
        if eng is not None:
            return eng.clock()
        return time.time()

    def client_host(request: Request) -> str:
        if request.client is None:
            return "unknown"
        return request.client.host

    # ---- browser side -------------------------------------------------------

    def browser_site(request: Request, site_key: str) -> SiteRecord:
        """The site for this key if the Origin header is one of its allowed origins; otherwise a uniform 403."""
        origin = request.headers.get("origin")
        site = store.get_site_by_key(site_key)
        if origin is None or site is None or site.disabled or origin not in site.origins:
            raise ApiError(403, "forbidden", "unknown site key or origin not allowed")
        request.scope["keyauth_cors_origin"] = origin
        return site

    def browser_rate(request: Request, endpoint: str, site: SiteRecord, user_id: str) -> None:
        wait = limiter.check([("ip", client_host(request)), ("v1user", site.site_id + ":" + user_id)])
        if wait is not None:
            logger.info("endpoint=%s site=%s user=%s outcome=rate_limited", endpoint, site.site_id, user_tag(user_id))
            raise ApiError(429, "rate_limited", "too many requests", {"Retry-After": str(wait)})

    @app.middleware("http")
    async def v1_cors(request: Request, call_next):
        response = await call_next(request)
        origin = request.scope.get("keyauth_cors_origin")
        if origin is not None:
            response.headers["Access-Control-Allow-Origin"] = origin  # the one verified origin, never "*"
            response.headers["Vary"] = "Origin"
        return response

    def preflight(request: Request) -> Response:
        # A preflight has no body, so the site is unknown: allow the origin if any enabled site lists it.
        # Every real request is still checked against its own site.
        origin = request.headers.get("origin")
        allowed = False
        if origin is not None:
            allowed = any(origin in s.origins for s in store.list_sites() if not s.disabled)
        if not allowed:
            raise ApiError(403, "forbidden", "origin not allowed")
        headers = {
            "Access-Control-Allow-Origin": origin,
            "Access-Control-Allow-Methods": "POST",
            "Access-Control-Allow-Headers": "Content-Type",
            "Access-Control-Max-Age": "600",
            "Vary": "Origin",
        }
        return Response(status_code=204, headers=headers)

    @router.options("/enroll")
    def enroll_preflight(request: Request):
        return preflight(request)

    @router.options("/verify")
    def verify_preflight(request: Request):
        return preflight(request)

    @router.post("/enroll")
    def enroll(body: V1EnrollIn, request: Request):
        site = browser_site(request, body.sitekey)
        browser_rate(request, "v1_enroll", site, body.user_id)
        digest = ticket_hash(body.ticket)
        if not store.ticket_valid(digest, site.site_id, body.user_id, now()):
            logger.info("endpoint=v1_enroll site=%s user=%s outcome=invalid_ticket", site.site_id, user_tag(body.user_id))
            raise ApiError(403, "invalid_ticket", "a valid enrollment ticket is required")
        eng = require_engine()
        timings = to_timings(body)
        result = eng.enroll(site.site_id, body.user_id, timings)
        outcome = result.status
        if result.reason is not None:
            outcome = result.reason
        logger.info("endpoint=v1_enroll site=%s user=%s outcome=%s", site.site_id, user_tag(body.user_id), outcome)
        if result.status == "complete":
            store.delete_ticket(digest)  # one ticket, one enrollment
        if result.status == "rejected" and result.reason == "already_enrolled":
            raise ApiError(409, "already_enrolled", "this user is already enrolled")
        if result.status == "rejected" and result.reason == "replay":
            raise ApiError(409, "replay", "this attempt is identical to an earlier one")
        if result.status == "rejected" and result.reason == "length_mismatch":
            raise ApiError(422, "length_mismatch", "attempt has a different number of keys than earlier samples")
        payload = {"status": result.status, "samples": result.samples, "required": result.required}
        if result.reason is not None:
            payload["reason"] = result.reason
        return payload

    @router.post("/verify")
    def verify(body: V1AttemptIn, request: Request):
        site = browser_site(request, body.sitekey)
        browser_rate(request, "v1_verify", site, body.user_id)
        eng = require_engine()
        timings = to_timings(body)
        result = eng.verify(site.site_id, body.user_id, timings)
        logger.info("endpoint=v1_verify site=%s user=%s outcome=%s", site.site_id, user_tag(body.user_id), result.reason)
        match = result.accepted
        risk = "high"
        if match and result.score is not None and result.score <= 0.5 * result.threshold:
            risk = "low"
        elif match:
            risk = "medium"
        issued = now()
        claims = TokenClaims(site.site_id, body.user_id, match, risk, issued, issued + TOKEN_TTL_S, new_jti())
        return {"token": sign_token(key, claims)}

    # ---- server side --------------------------------------------------------

    def server_guard(request: Request, secret: str) -> None:
        """Browsers must not be able to call these; then rate limit per IP and per claimed site."""
        if "origin" in request.headers:
            raise ApiError(403, "forbidden", "server endpoint")
        keys: list = [("ip", client_host(request))]
        claimed = secret_site_id(secret)
        if claimed is not None and store.get_site(claimed) is not None:
            keys.append(("site", claimed))
        wait = server_limiter.check(keys)
        if wait is not None:
            raise ApiError(429, "rate_limited", "too many requests", {"Retry-After": str(wait)})

    def server_site(secret: str) -> SiteRecord:
        site = authenticate_secret(store, secret)
        if site is None:
            raise ApiError(401, "unauthorized", "invalid secret")
        return site

    @router.post("/tickets")
    def tickets(body: TicketIn, request: Request):
        server_guard(request, body.secret)
        site = server_site(body.secret)
        user = store.get_user(site.site_id, body.user_id)
        if user is not None and user.status == "active" and not body.reset:
            raise ApiError(409, "already_enrolled", "user is enrolled; pass reset=true to enroll again")
        if user is not None and user.status == "active":
            store.delete_user(site.site_id, body.user_id)
        ticket = secrets.token_urlsafe(32)
        current = now()
        store.add_ticket(ticket_hash(ticket), site.site_id, body.user_id, current + TICKET_TTL_S, current)
        logger.info("endpoint=v1_tickets site=%s user=%s outcome=issued reset=%s", site.site_id, user_tag(body.user_id), body.reset)
        return {"ticket": ticket, "expires_in": int(TICKET_TTL_S)}

    @router.post("/siteverify")
    def siteverify(body: SiteverifyIn, request: Request):
        server_guard(request, body.secret)
        site = server_site(body.secret)
        current = now()
        claims = verify_token(key, body.token, current)
        # the token is only consumed when it is genuine, unexpired and minted for THIS site, so a
        # wrong-site presentation cannot burn someone else's token
        if claims is None or claims.site_id != site.site_id or not store.consume_token(claims.jti, claims.expires_at, current):
            logger.info("endpoint=v1_siteverify site=%s outcome=invalid_token", site.site_id)
            return {"success": False, "error": "invalid_token"}
        logger.info("endpoint=v1_siteverify site=%s user=%s outcome=ok", site.site_id, user_tag(claims.user_id))
        return {"success": True, "user_id": claims.user_id, "rhythm_match": claims.rhythm_match, "risk": claims.risk}

    app.include_router(router)
