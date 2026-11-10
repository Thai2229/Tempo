"""FastAPI app factory. Behavior and error mapping are documented in keyauth.api."""

from __future__ import annotations

import hashlib
import hmac
import logging
import math
import threading
from contextlib import asynccontextmanager
from typing import Annotated, Literal

import numpy as np
from fastapi import FastAPI, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, model_validator
from starlette.exceptions import HTTPException as StarletteHTTPException

from keyauth.api.ratelimit import SlidingWindowLimiter
from keyauth.api.settings import Settings
from keyauth.auth import EngineConfig, InvalidTimings, KeyAuthEngine, TemplateStore, validate_timings
from keyauth.config import MAX_KEYS
from keyauth.events import RawEvent, pair_events
from keyauth.features import extract_features

logger = logging.getLogger("keyauth.api")

MAX_EVENTS = 4 * MAX_KEYS
MAX_BODY_BYTES = 128 * 1024
USER_ID_PATTERN = r"^[A-Za-z0-9._@-]{1,64}$"

UserId = Annotated[str, Field(min_length=1, max_length=64, pattern=USER_ID_PATTERN)]


class EventIn(BaseModel):
    key: str = Field(min_length=1, max_length=32)
    type: Literal["down", "up"]
    t: float = Field(allow_inf_nan=False)  # milliseconds (performance.now())
    code: str = Field(default="", max_length=32)


class AttemptIn(BaseModel):
    user_id: UserId
    events: list[EventIn] | None = Field(default=None, max_length=MAX_EVENTS)
    timings: list[list[float]] | None = Field(default=None, max_length=MAX_KEYS)  # seconds

    @model_validator(mode="after")
    def exactly_one_input(self) -> "AttemptIn":
        if (self.events is None) == (self.timings is None):
            raise ValueError("send exactly one of 'events' or 'timings'")
        return self


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str, headers: dict[str, str] | None = None):
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message
        self.headers = headers


def error_response(status: int, code: str, message: str, headers: dict[str, str] | None = None) -> JSONResponse:
    return JSONResponse({"error": {"code": code, "message": message}}, status_code=status, headers=headers)


def user_tag(user_id: str) -> str:
    return hashlib.sha256(user_id.encode()).hexdigest()[:12]


def to_timings(body: AttemptIn) -> np.ndarray:
    """Request body -> validated (n, 3) float32 timings in seconds."""
    if body.timings is not None:
        return validate_timings(body.timings)
    try:
        raw = [RawEvent(key=e.key, type=e.type, t=e.t, code=e.code) for e in body.events]
        keystrokes = pair_events(raw)
        values = extract_features(keystrokes, check_bounds=False).values
    except ValueError as e:
        raise InvalidTimings(str(e)) from None
    return validate_timings(values)


def build_engine(settings: Settings) -> KeyAuthEngine:
    from keyauth.auth.embedder import TorchEmbedder  # imports torch, so only when needed

    embedder, threshold = TorchEmbedder.from_checkpoint(settings.checkpoint)
    per_user = None
    if settings.per_user:
        per_user = TorchEmbedder.per_user_params(settings.checkpoint)
    return KeyAuthEngine(embedder, TemplateStore(settings.db_path), threshold, EngineConfig(), per_user=per_user)


def create_app(engine: KeyAuthEngine | None = None, settings: Settings | None = None) -> FastAPI:
    if settings is None:
        settings = Settings.from_env()
    state = {"engine": engine, "attempted": engine is not None}
    load_lock = threading.Lock()
    limiter = SlidingWindowLimiter(settings.rate_limit, settings.rate_window_s)  # tests swap app.state.limiter.clock

    def get_engine() -> KeyAuthEngine | None:
        with load_lock:
            if not state["attempted"]:
                state["attempted"] = True
                try:
                    state["engine"] = build_engine(settings)
                except Exception as e:  # missing/corrupt checkpoint, no torch, unwritable db
                    logger.error("model load failed: %s", type(e).__name__)
        return state["engine"]

    def require_engine() -> KeyAuthEngine:
        eng = get_engine()
        if eng is None:
            raise ApiError(503, "model_unavailable", "the model is not loaded")
        return eng

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        get_engine()
        yield

    app = FastAPI(title="keyauth", lifespan=lifespan)
    app.state.settings = settings
    app.state.limiter = limiter

    # ---- errors -----------------------------------------------------------

    @app.exception_handler(ApiError)
    async def api_error_handler(_: Request, exc: ApiError):
        return error_response(exc.status, exc.code, exc.message, exc.headers)

    @app.exception_handler(RequestValidationError)
    async def validation_handler(_: Request, exc: RequestValidationError):
        # loc + msg only: pydantic's "input" would echo the submitted timings
        details = [{"loc": [str(p) for p in err["loc"]], "msg": str(err["msg"])} for err in exc.errors()]
        body = {"error": {"code": "validation_error", "message": "invalid request", "details": details}}
        return JSONResponse(body, status_code=422)

    @app.exception_handler(InvalidTimings)
    async def invalid_timings_handler(_: Request, exc: InvalidTimings):
        return error_response(422, "invalid_timings", str(exc))

    @app.exception_handler(StarletteHTTPException)
    async def http_error_handler(_: Request, exc: StarletteHTTPException):
        codes = {404: "not_found", 405: "method_not_allowed"}
        return error_response(exc.status_code, codes.get(exc.status_code, "http_error"), str(exc.detail), exc.headers)

    @app.exception_handler(Exception)
    async def unexpected_handler(_: Request, exc: Exception):
        logger.error("unhandled %s", type(exc).__name__)
        return error_response(500, "internal_error", "internal error")

    @app.middleware("http")
    async def limit_body_size(request: Request, call_next):
        length = request.headers.get("content-length", "0")
        if length.isdigit() and int(length) > MAX_BODY_BYTES:
            return error_response(413, "payload_too_large", f"body must be at most {MAX_BODY_BYTES} bytes")
        return await call_next(request)

    # ---- helpers ----------------------------------------------------------

    def check_rate(request: Request, endpoint: str, user_id: str) -> None:
        host = "unknown"
        if request.client is not None:
            host = request.client.host
        wait = limiter.check([("ip", host), ("user", user_id)])
        if wait is not None:
            logger.info("endpoint=%s user=%s outcome=rate_limited", endpoint, user_tag(user_id))
            raise ApiError(429, "rate_limited", "too many requests", {"Retry-After": str(wait)})

    def retry_after(eng: KeyAuthEngine, user_id: str) -> int:
        """Whole seconds (>= 1) until the user's lockout ends, 0 if not locked."""
        user = eng.store.get_user(settings.app_id, user_id)
        if user is None:
            return 0
        left = user.locked_until - eng.clock()
        if left <= 0:
            return 0
        return max(1, math.ceil(left))

    # ---- endpoints --------------------------------------------------------

    @app.get("/health")
    def health():
        return {"status": "ok", "model_loaded": get_engine() is not None}

    @app.post("/enroll")
    def enroll(body: AttemptIn, request: Request):
        check_rate(request, "enroll", body.user_id)
        eng = require_engine()
        timings = to_timings(body)
        result = eng.enroll(settings.app_id, body.user_id, timings)
        outcome = result.status
        if result.reason is not None:
            outcome = result.reason
        logger.info("endpoint=enroll user=%s outcome=%s", user_tag(body.user_id), outcome)
        if result.status == "rejected" and result.reason == "already_enrolled":
            raise ApiError(409, "already_enrolled", "this user is already enrolled")
        if result.status == "rejected" and result.reason == "replay":
            raise ApiError(409, "replay", "this attempt is identical to an earlier one")
        if result.status == "rejected" and result.reason == "length_mismatch":
            raise ApiError(422, "length_mismatch", "attempt has a different number of keys than earlier samples")
        payload = {"status": result.status, "samples": result.samples, "required": result.required}
        if result.reason is not None:
            payload["reason"] = result.reason  # "inconsistent": enrollment restarted
        return payload

    @app.post("/verify")
    def verify(body: AttemptIn, request: Request):
        check_rate(request, "verify", body.user_id)
        eng = require_engine()
        timings = to_timings(body)
        result = eng.verify(settings.app_id, body.user_id, timings)
        logger.info("endpoint=verify user=%s outcome=%s", user_tag(body.user_id), result.reason)
        if result.reason == "not_enrolled":
            raise ApiError(404, "not_enrolled", "user is not enrolled")
        if result.reason == "locked":
            wait = max(1, retry_after(eng, body.user_id))
            raise ApiError(423, "locked", "too many failed attempts, try again later", {"Retry-After": str(wait)})
        if result.reason == "replay":
            raise ApiError(409, "replay", "this attempt is identical to an earlier one")
        if result.reason == "length_mismatch":
            raise ApiError(422, "length_mismatch", "attempt has a different number of keys than at enrollment")
        if result.reason == "scorer_error":
            raise ApiError(503, "scorer_error", "stored template is unusable")
        payload: dict = {"accepted": result.accepted, "reason": result.reason}
        if not result.accepted:
            wait = retry_after(eng, body.user_id)
            if wait > 0:
                payload["retry_after_s"] = wait
        if settings.debug_scores:
            payload["score"] = result.score
            payload["threshold"] = result.threshold
        return payload

    @app.get("/status")
    def status(user_id: Annotated[str, Query(min_length=1, max_length=64, pattern=USER_ID_PATTERN)]):
        info = require_engine().status(settings.app_id, user_id)
        if info is None:
            raise ApiError(404, "not_enrolled", "user is not enrolled")
        return info

    @app.delete("/user")
    def delete_user(request: Request, user_id: Annotated[str, Query(min_length=1, max_length=64, pattern=USER_ID_PATTERN)]):
        if settings.admin_token is not None:
            given = request.headers.get("authorization", "")
            if not hmac.compare_digest(given.encode(), f"Bearer {settings.admin_token}".encode()):
                raise ApiError(401, "unauthorized", "admin token required", {"WWW-Authenticate": "Bearer"})
        deleted = require_engine().delete_user(settings.app_id, user_id)
        logger.info("endpoint=delete user=%s outcome=%s", user_tag(user_id), deleted)
        if not deleted:
            raise ApiError(404, "not_enrolled", "user is not enrolled")
        return {"deleted": True}

    return app

