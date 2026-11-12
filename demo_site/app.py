"""The demo "customer" site. Pages: /register, /login, /account. Policy and design notes: demo_site/README.md.

Login order (the order matters, and tests/test_demo_site.py pins it):
  1. check the password hash (scrypt, constant time); wrong or unknown user -> one generic error, tempo is NOT called
  2. only then call tempo /v1/siteverify with the token from the form
  3. policy (the customer's choice, see decide_login): match for THIS account -> in; no match -> second step;
     no token / tempo unreachable -> fail-open to password only, flagged in the log (or second step if fail_mode=closed)
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles

from demo_site import pages
from demo_site.config import SiteConfig
from demo_site.tempo_client import TempoClient, TempoLike
from demo_site.store import Session, Store, User, check_password, hash_password

logger = logging.getLogger("demo_site")

COOKIE = "demo_session"
ACTIVE_TTL_S = 3600.0
PENDING_TTL_S = 600.0
MAX_CODE_ATTEMPTS = 5
MAX_TOKEN_CHARS = 2048  # tempo's own limit; longer is garbage, not "tempo is down"
USERNAME_CHARS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._@+-")
GENERIC_LOGIN_ERROR = "Invalid username or password."

ALLOW = "allow"
ALLOW_FLAGGED = "allow_flagged"
SECOND_STEP = "second_step"


@dataclass(frozen=True)
class Decision:
    action: str
    reason: str


def decide_login(fail_mode: str, token: str, verdict: dict[str, Any] | None, account_opaque_id: str) -> Decision:
    """The policy. Called only AFTER the password was verified. This is the customer's choice, not tempo's.

    token   what the form carried ("" = tempo.js did not run, was blocked, or tempo did not answer in the browser)
    verdict the parsed /v1/siteverify answer, or None when that call could not be completed (only used with a token)
    """
    unavailable = ""
    if token == "":
        unavailable = "no_token"
    elif len(token) <= MAX_TOKEN_CHARS and verdict is None:
        unavailable = "tempo_unreachable"
    if unavailable != "":
        if fail_mode == "closed":
            return Decision(SECOND_STEP, unavailable + "_fail_closed")
        return Decision(ALLOW_FLAGGED, unavailable + "_fail_open")
    if len(token) > MAX_TOKEN_CHARS or verdict is None or verdict.get("success") is not True:
        return Decision(SECOND_STEP, "invalid_token")
    if verdict.get("user_id") != account_opaque_id:
        return Decision(SECOND_STEP, "token_for_other_user")
    if verdict.get("rhythm_match") is True:
        return Decision(ALLOW, "rhythm_match risk=" + str(verdict.get("risk")))
    return Decision(SECOND_STEP, "rhythm_mismatch")


def security_headers(config: SiteConfig) -> dict[str, str]:
    ka = config.tempo_url
    csp = (
        f"default-src 'none'; script-src 'self' {ka}; connect-src 'self' {ka}; style-src 'self'; img-src 'self'; "
        "form-action 'self'; base-uri 'none'; frame-ancestors 'none'"
    )
    return {
        "Content-Security-Policy": csp,
        "X-Content-Type-Options": "nosniff",
        "Referrer-Policy": "no-referrer",
        "Cache-Control": "no-store",
    }


def form_fields(body: bytes) -> dict[str, str]:
    """application/x-www-form-urlencoded -> first value of each field (no multipart dependency needed)."""
    try:
        parsed = parse_qs(body.decode("utf-8", errors="replace"), keep_blank_values=True, max_num_fields=20)
    except ValueError:
        return {}
    return {k: v[0] for k, v in parsed.items()}


def valid_username(name: str) -> bool:
    return 3 <= len(name) <= 64 and all(c in USERNAME_CHARS for c in name)


def valid_password(password: str) -> bool:
    return 8 <= len(password) <= 100


def create_app(config: SiteConfig, client: TempoLike | None = None, store: Store | None = None) -> FastAPI:
    if client is None:
        client = TempoClient(config.tempo_url, config.site_secret)
    if store is None:
        store = Store(config.db_path)
    headers = security_headers(config)
    dummy_hash = hash_password(secrets.token_hex(16))  # verified against for unknown usernames: same work either way
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    app.mount("/static", StaticFiles(directory=str(Path(__file__).parent / "static")), name="static")

    @app.middleware("http")
    async def add_headers(request: Request, call_next):
        response = await call_next(request)
        for name, value in headers.items():
            response.headers.setdefault(name, value)
        return response

    def html(title: str, body: str, scripts: list[str], user: str | None = None, status: int = 200) -> HTMLResponse:
        return HTMLResponse(pages.layout(title, body, config.site_key, config.tempo_url, scripts, user), status_code=status)

    def set_cookie(response: Response, sid: str, ttl_s: float) -> None:
        response.set_cookie(COOKIE, sid, max_age=int(ttl_s), path="/", httponly=True, samesite="lax", secure=config.secure_cookie)

    def current(request: Request) -> Session | None:
        return store.get_session(request.cookies.get(COOKIE))

    def start_session(request: Request, response: Response, user: User, state: str, code_hash: str | None = None, method: str = "") -> None:
        store.delete_session(request.cookies.get(COOKIE))  # always a fresh id: no session fixation
        ttl = ACTIVE_TTL_S
        if state == "pending":
            ttl = PENDING_TTL_S
        set_cookie(response, store.create_session(user, state, ttl, code_hash, method), ttl)

    def stand_in_id(username: str) -> str:
        """Same look as a real opaque id (u_ + 32 hex), stable per username, unpredictable without the pepper."""
        return "u_" + hmac.new(config.pepper, username.strip().lower().encode(), hashlib.sha256).hexdigest()[:32]

    # ---- pages ---------------------------------------------------------------

    @app.get("/")
    def index(request: Request):
        session = current(request)
        if session is not None and session.state == "active":
            return RedirectResponse("/account", status_code=303)
        return RedirectResponse("/login", status_code=303)

    @app.get("/register")
    def register_page():
        return html("Register", pages.register_form(), [])

    @app.post("/register")
    async def register(request: Request):
        fields = form_fields(await request.body())
        username = fields.get("username", "").strip()
        password = fields.get("password", "")
        if not valid_username(username) or not valid_password(password):
            return html("Register", pages.register_form("Username: 3 to 64 letters, digits or . _ @ + -. Password: 8 to 100 characters."), [], status=422)
        user = store.create_user(username, password)
        if user is None:
            # registration has to say so; the LOGIN path never reveals whether a username exists
            return html("Register", pages.register_form("That username is taken."), [], status=409)
        logger.info("register user_id=%s", user.opaque_id)
        response = RedirectResponse("/account", status_code=303)
        start_session(request, response, user, "active", method="registered")
        return response

    @app.get("/login")
    def login_page():
        return html("Log in", pages.login_form(), ["login.js"])

    @app.post("/login/identify")
    async def identify(request: Request):
        """Which opaque id should tempo.js ask about? Same answer shape for every username, so this does not reveal
        whether the account exists: unknown names get a stable stand-in id that tempo treats as "unknown user"."""
        try:
            data = await request.json()
        except ValueError:
            data = None
        username = ""
        if isinstance(data, dict) and isinstance(data.get("username"), str):
            username = data["username"].strip()
        if username == "" or len(username) > 254:
            return JSONResponse({"error": "bad_request"}, status_code=400)
        user = store.get_user(username)
        user_id = stand_in_id(username)
        if user is not None:
            user_id = user.opaque_id
        return JSONResponse({"user_id": user_id})

    @app.post("/login")
    async def login(request: Request):
        fields = form_fields(await request.body())
        username = fields.get("username", "").strip()
        password = fields.get("password", "")
        token = fields.get("tempo_token", "")
        user = store.get_user(username)
        stored = dummy_hash
        if user is not None:
            stored = user.password_hash
        password_ok = check_password(password, stored)  # always runs scrypt, also for unknown names
        if user is None or not password_ok:
            # tempo is NOT contacted: a wrong password must never consume or reveal anything
            logger.info("login outcome=bad_credentials")
            return html("Log in", pages.login_form(GENERIC_LOGIN_ERROR), ["login.js"], status=401)
        verdict = None
        if token != "" and len(token) <= MAX_TOKEN_CHARS:
            verdict = client.siteverify(token)
        decision = decide_login(config.fail_mode, token, verdict, user.opaque_id)
        if decision.action == ALLOW_FLAGGED:
            logger.warning("FLAG login without rhythm check (password only) user_id=%s reason=%s", user.opaque_id, decision.reason)
        else:
            logger.info("login user_id=%s decision=%s reason=%s", user.opaque_id, decision.action, decision.reason)
        if decision.action == SECOND_STEP:
            code = "{:06d}".format(secrets.randbelow(1000000))
            logger.warning("SECOND-STEP CODE (stub for an email) user_id=%s code=%s", user.opaque_id, code)
            response = RedirectResponse("/login/confirm", status_code=303)
            start_session(request, response, user, "pending", hashlib.sha256(code.encode()).hexdigest())
            return response
        method = "password_only"
        if decision.action == ALLOW:
            method = "rhythm"
        response = RedirectResponse("/account", status_code=303)
        start_session(request, response, user, "active", method=method)
        return response

    @app.get("/login/confirm")
    def confirm_page(request: Request):
        session = current(request)
        if session is None or session.state != "pending":
            return RedirectResponse("/login", status_code=303)
        return html("Confirm", pages.confirm_form(), [])

    @app.post("/login/confirm")
    async def confirm(request: Request):
        sid = request.cookies.get(COOKIE)
        session = store.get_session(sid)
        if session is None or session.state != "pending" or session.code_hash is None or sid is None:
            return RedirectResponse("/login", status_code=303)
        if session.attempts >= MAX_CODE_ATTEMPTS:
            store.delete_session(sid)
            return RedirectResponse("/login", status_code=303)
        store.count_attempt(sid)
        code = form_fields(await request.body()).get("code", "").strip()
        given = hashlib.sha256(code.encode()).hexdigest()
        if not hmac.compare_digest(given, session.code_hash):
            logger.info("second step outcome=wrong_code user_id=%s", session.user.opaque_id)
            return html("Confirm", pages.confirm_form("That code is not right."), [], status=401)
        logger.info("second step outcome=confirmed user_id=%s", session.user.opaque_id)
        response = RedirectResponse("/account", status_code=303)
        start_session(request, response, session.user, "active", method="code")
        return response

    @app.get("/account")
    def account(request: Request):
        session = current(request)
        if session is None or session.state != "active":
            return RedirectResponse("/login", status_code=303)
        return html("Account", pages.account_body(session.method), ["account.js"], user=session.user.username)

    @app.post("/account/ticket")
    async def ticket(request: Request):
        """Enrollment ticket for the LOGGED-IN user only (the id comes from the server-side session, never from the request)."""
        session = current(request)
        if session is None or session.state != "active":
            return JSONResponse({"error": "login_required"}, status_code=401)
        try:
            data = await request.json()
        except ValueError:
            data = None
        reset = isinstance(data, dict) and data.get("reset") is True
        status, body = client.create_ticket(session.user.opaque_id, reset)
        if status == 200 and body is not None and isinstance(body.get("ticket"), str):
            logger.info("ticket issued user_id=%s reset=%s", session.user.opaque_id, reset)
            return JSONResponse({"ticket": body["ticket"], "user_id": session.user.opaque_id, "sitekey": config.site_key})
        if status == 409:
            return JSONResponse({"error": "already_enrolled"}, status_code=409)
        logger.error("ticket failed http_status=%s", status)
        return JSONResponse({"error": "tempo_unavailable"}, status_code=502)

    @app.post("/logout")
    def logout(request: Request):
        store.delete_session(request.cookies.get(COOKIE))
        response = RedirectResponse("/login", status_code=303)
        response.delete_cookie(COOKIE, path="/")
        return response

    return app
