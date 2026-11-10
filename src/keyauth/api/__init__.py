"""HTTP API for keyauth (FastAPI).

Run:  python -m keyauth.api          (or: keyauth-demo, or: uvicorn keyauth.api:app)
Demo: the command above also serves the web page at http://127.0.0.1:8000/ (KEYAUTH_SERVE_UI=0 turns
      it off). Without a checkpoint it still starts, the page shows "model not loaded", and a hint
      is printed: python scripts/get_cmu.py && python -m keyauth.train --data cmu
Config: environment variables, see .env.example. Needs `pip install -e ".[app]"`.

The module-level `app` never fails on import: if the checkpoint cannot be loaded,
GET /health reports model_loaded=false and /enroll, /verify answer 503 "model_unavailable".

Endpoints
  POST   /enroll         {user_id, events | timings}
  POST   /verify         {user_id, events | timings}
  GET    /status?user_id=
  DELETE /user?user_id=   (Authorization: Bearer <KEYAUTH_ADMIN_TOKEN> if that variable is set)
  GET    /health
  GET    /, /static/<file>   demo page and its assets (HTML carries a strict CSP; 404 JSON when disabled)

Exactly one of `events` (raw keydown/keyup: [{key, type: "down"|"up", t in ms, code?}], as the
browser captures them with performance.now()) or `timings` ([[hold, dd, ud], ...] in seconds) is
required. Events are paired and converted with keyauth.events / keyauth.features, then both paths
go through keyauth.auth.validation.validate_timings.

app_id is Settings.app_id (KEYAUTH_APP_ID): a single-tenant demo, clients cannot pick it.
The client IP for rate limiting is the socket peer (X-Forwarded-For is NOT trusted).

Errors always look like {"error": {"code": ..., "message": ...}}:
  422 validation_error     malformed body / query (incl. bad user_id, neither or both of events|timings,
                           too many events)
  422 invalid_timings      events or timings rejected by validation (bad pairing, out of range, ...)
  413 payload_too_large    Content-Length above 128 KiB
  404 not_enrolled         unknown user (verify, status, delete) or enrollment not finished (verify)
  423 locked               engine lockout after max_failures; Retry-After = seconds left
  429 rate_limited         per-IP or per-user sliding window on /enroll and /verify; Retry-After set
  409 replay               an identical attempt was already seen (enroll and verify)
  409 already_enrolled     /enroll for a user who finished enrollment
  422 length_mismatch      different number of keys than at enrollment (enroll and verify)
  503 model_unavailable    model/checkpoint not loaded
  503 scorer_error         stored template unusable; fails closed
  401 unauthorized         DELETE /user without the admin token (only if one is configured)
  500 internal_error       anything unexpected (no details)
A verify that merely does not match is a normal 200 {"accepted": false, "reason": "no_match"}.
Replay / length_mismatch / locked on /verify are never accepted and the first two count as failures
toward the lockout. 200 /verify responses add "retry_after_s" when that attempt triggered the lockout.
score and threshold are returned by /verify only when KEYAUTH_DEBUG_SCORES=1.

Logs carry a short hash of user_id, the endpoint and the outcome reason only; never timings,
events, embeddings or scores.

Rate limits are in memory and per process: with several workers each has its own counters.
DELETE without KEYAUTH_ADMIN_TOKEN is open (demo only): set the token anywhere else, since
delete + re-enroll replaces a user's template.
"""

from keyauth.api.app import create_app
from keyauth.api.settings import Settings

app = create_app()

__all__ = ["Settings", "app", "create_app"]
