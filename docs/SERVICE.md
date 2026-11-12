# Multi-site service (C12)

A Turnstile-style service. A customer site embeds our script (C13); the visitor's browser sends typing timings to us
and gets back only an opaque signed token. The customer's own server then calls `/v1/siteverify` and decides. The
browser never learns "accepted", so it cannot be used as an oracle to tune a forgery.

This is a portfolio demo, not a hardened production service. The Aalto non-commercial license limit does not apply
to the shipped CMU-trained model, but the demo caveat in the README stands.

## Setup
```
export TEMPO_ENABLE_V1=1
export TEMPO_TOKEN_KEY=$(python -c "import secrets; print(secrets.token_urlsafe(48))")   # 32+ chars; no default
python -m tempo.sites create --name "My shop" --origin https://shop.example   # prints site key + secret ONCE
python -m tempo.sites list | disable SITE_ID | enable SITE_ID | rotate-secret SITE_ID
python -m tempo.api
```
With `TEMPO_ENABLE_V1=1` the app refuses to start if `TEMPO_TOKEN_KEY` is missing or shorter than 32 characters.
With it off (default) `/v1` does not exist and the single-tenant demo API is unchanged. Rotating the signing key
invalidates all tokens in flight (at most 60 s of them). Run the CLI against the same `TEMPO_DB` as the server.

## Flow
1. Enrollment: the customer's server authenticates the user itself, then `POST /v1/tickets {secret, user_id}`
   and hands the ticket to the page. The page calls `POST /v1/enroll` (10 attempts, same ticket) with the site key.
   The ticket is used up when enrollment completes; it also expires after 10 minutes.
2. Login: the page calls `POST /v1/verify {sitekey, user_id, events|timings}` and puts the returned `token` in a
   hidden form field next to the password. The customer's server checks the password, then
   `POST /v1/siteverify {secret, token}` and compares the returned `user_id` with the account that just logged in.
3. The customer decides: `rhythm_match`, `risk` (`low` / `medium` / `high`), e.g. require 2FA on `false`.

## Endpoints
| Endpoint | Caller | Auth | Notes |
|---|---|---|---|
| `POST /v1/enroll` `{sitekey, user_id, ticket, events\|timings}` | browser | site key + Origin + ticket | `{status, samples, required}` |
| `POST /v1/verify` `{sitekey, user_id, events\|timings}` | browser | site key + Origin | always `{token}` |
| `OPTIONS /v1/enroll`, `/v1/verify` | browser | Origin | preflight, 204 |
| `POST /v1/tickets` `{secret, user_id, reset?}` | customer server | secret | `{ticket, expires_in}`; 409 if enrolled and no `reset` |
| `POST /v1/siteverify` `{secret, token}` | customer server | secret | `{success, user_id, rhythm_match, risk}` or `{success: false, error: "invalid_token"}` |

Errors use `{"error": {"code", "message"}}`: 403 `forbidden` (unknown or disabled site key, missing or other Origin: all
the same answer), 403 `invalid_ticket`, 401 `unauthorized` (bad secret), 422, 429 (`Retry-After`), 503.

Decisions:
- `siteverify` identifies the site from the secret alone: secrets look like `sk.<site_id>.<random>`, and the secret is
  checked against that site's salted SHA-256 (constant-time compare). The site key is not needed there.
- `/v1/verify` answers `200 {token}` for match, mismatch, unknown user, replay, wrong length and locked-out users.
  Lockout is therefore not a 423 on `/v1`: it would tell the browser which ids exist and are under attack. The
  customer sees `rhythm_match=false, risk="high"`. The engine's lockout, replay and length checks still run. 422
  (malformed input) and 429 (rate limit, per IP and per site+user) do not depend on a user's state and are kept.
- `risk`: `high` = no match; `low` = match with score at most half the threshold; `medium` = other matches.
- `reset=true` on `/v1/tickets` deletes the user's template at issue time (the server is authenticated), so a
  reset needs a new ticket and a new enrollment. A new ticket for the same user replaces the old one.
- CORS: browser endpoints echo the one verified Origin (never `*`), with `Vary: Origin`. A preflight has no body,
  so it is allowed if any enabled site lists the origin; every real request is checked against its own site.
  Server endpoints send no CORS headers, answer 405 to OPTIONS and refuse any request carrying an `Origin` header
  (403), so a page cannot even try to use a leaked secret from a browser.
- User ids: `[A-Za-z0-9._-]{1,64}`. Opaque ids chosen by the customer; do not use emails (`@` is rejected).
  Logs carry only a short hash of the id, the site id and the outcome; never secrets, tickets, tokens or timings.

## Embedding
Served by the service itself (only when `TEMPO_ENABLE_V1=1`): `GET /v1/tempo.js`, `Content-Type: application/javascript`,
`Cache-Control: public, max-age=300`, `nosniff`, and `Access-Control-Allow-Origin: *` (this one public static file only).
```html
<form method="post" action="/login">
  <input id="username" name="username" autocomplete="username">
  <input id="pw" name="password" type="password" autocomplete="current-password">
  <button>Log in</button>
</form>
<script src="https://TEMPO-HOST/v1/tempo.js" defer></script>
<script src="/login-page.js" defer></script>  <!-- same-origin file; no inline script needed -->
```
```js
// login-page.js
const k = tempo.attach(document.getElementById("pw"), {sitekey: "pk...", userIdField: "#username"});
// the form now posts an extra hidden field "tempo_token"; k.destroy() / k.reset() / await k.getToken() also exist
```
Enrollment (the page repeats it `required` times; the ticket comes from your server via `/v1/tickets`):
```js
const {samples, required, done} = await tempo.enroll(pwInput, {sitekey, ticket, user_id});   // one typed attempt per call
```
API:
- `tempo.attach(input, {sitekey, endpoint?, tokenField? = "tempo_token", userIdField?, mode? = "verify"})` returns `{destroy(), reset(), getToken()}`.
  `endpoint` defaults to the origin the script was loaded from. `userIdField` is a selector, an element or a function
  returning the opaque id (`[A-Za-z0-9._-]{1,64}`, no emails). `getToken()` resolves to the token string, or `""`; it never rejects.
  Attaching twice to one input replaces the first instance.
- `tempo.enroll(input, {sitekey, ticket, user_id, endpoint?})` resolves `{samples, required, done}` (plus `reason` when the
  server restarted enrollment, e.g. `"inconsistent"`) and rejects with an `Error` whose `.code` is the server's error code
  (`invalid_ticket`, `replay`, `length_mismatch`, `rate_limited`, ...) or `network` / `timeout` / `cancelled` / `bad_arguments`.
  Corrections and pastes are ignored: it waits for one clean attempt. Clear the field between samples.

Behaviour and decisions:
- The user must finish with Enter in the password field: Enter is part of the captured rhythm (as in enrollment), so the script
  holds back the Enter keydown, and submits the form after the Enter keyup (`form.requestSubmit()`). A click on the submit
  button without Enter, a paste, an autofill, Backspace/Delete, a blur, or a password shorter than 4 or longer than 100
  characters produces no attempt: the form is submitted unchanged with an EMPTY token field.
- It never reads `input.value`, and never sends or stores the password or key names. Events are anonymized in the browser before
  sending (every key but modifiers and Enter becomes `"x"`, codes become `k0`, `k1`, ...). Same code as the demo page: the logic of
  `capture.js` is copied into `tempo.js` between markers by `scripts/sync_tempo.py`; `pytest` fails if they drift
  (edit `capture.js`, run `python scripts/sync_tempo.py`). No `eval`, inline handlers, storage or cookies; the only request is
  `POST {endpoint}/v1/verify` (and `/v1/enroll`), sent without credentials. Because the Origin must be allowed for the site
  key (see CORS above), the customer page's CSP needs `script-src` and `connect-src` to include the tempo host.
- Fail open or closed is the CUSTOMER's choice. If tempo is unreachable, errors, or does not answer within 5 s, the form is still
  submitted with an empty token, so tempo can never lock a user out of your login. Your server decides what an empty token means:
  treating it like `rhythm_match=false` (for example asking for a second factor) is fail-closed; accepting the password alone is fail-open.
  Always do the `siteverify` call on your server, compare the returned `user_id` with the account that logged in, and
  remember the token is single use and lives 60 s.
- `mode` is accepted for future use; only `"verify"` exists.

## Token format
`base64url(payload) "." base64url(HMAC-SHA256(TEMPO_TOKEN_KEY, "tempo-token-v1." + base64url(payload)))`
with payload `{v:1, sid, uid, m (bool), r, iat, exp, jti}`, `exp - iat = 60 s`. There is no algorithm field, so there
is nothing to downgrade. Single use: `siteverify` inserts `jti` into a primary-keyed table (atomic; expired rows are
purged on each insert). The token is consumed only when it is genuine, unexpired and minted for the calling site, so
another site cannot burn it.

## Threat model: what each failure test defends
| Attack | Defense | Tests |
|---|---|---|
| Forged / edited token (bad signature, flipped `m`, other `uid`/`sid`/`exp`, `alg: none`-style, empty or foreign key, missing MAC) | HMAC checked in constant time before parsing; fixed algorithm; strict claim types | `test_token_roundtrip_and_bad_signature`, `test_altered_payload_*`, `test_alg_confusion_*`, `test_malformed_claims_*`, `test_forged_token_over_http` |
| Expired token | `exp` checked against an injectable clock | `test_expired_token`, `test_token_valid_until_expiry` |
| Reuse, including two parallel calls | atomic primary-key insert | `test_reused_token`, `test_concurrent_double_consume`, `test_store_consume_is_atomic_and_cleans_up` |
| Which check failed (oracle for the customer-side caller) | one `invalid_token` answer | `test_failure_reasons_are_indistinguishable` |
| Wrong / rotated secret, unknown or disabled site | 401, salted hash, constant-time compare | `test_wrong_secret`, `test_unknown_and_disabled_site`, `test_rotate_secret_invalidates_old` |
| Site B presenting site A's token | `sid` must equal the authenticated site; no burn on mismatch | `test_token_for_site_a_with_site_b_secret`, `test_templates_are_per_site` |
| Token for X used as Y | `user_id` comes from the signed token, never from the request | `test_token_for_user_x_is_x` |
| Browser learns the result | uniform `{token}` for every outcome | `test_browser_response_is_uniform`, `test_impostor_gets_a_token_*`, `test_lockout_is_a_token_not_423` |
| Enrolling or re-enrolling a victim from the browser | ticket bound to site + user, 10 min, used up by one enrollment | `test_enroll_without_or_with_bad_ticket`, `test_ticket_for_x_cannot_enroll_y`, `test_ticket_for_site_a_*`, `test_expired_ticket`, `test_ticket_is_used_up_*`, `test_victim_cannot_be_re_enrolled_*`, `test_new_ticket_replaces_old` |
| Calls from other origins; secrets used from a browser | per-site Origin check; server endpoints refuse Origin | `test_wrong_and_missing_origin`, `test_preflight`, `test_server_endpoints_are_not_cross_origin` |
| Secret leaks via the database or logs | only salted hash stored; tickets stored hashed; no secrets in logs | `test_secret_never_stored_in_plaintext`, `test_secret_not_in_logs`, `test_cli_prints_secret_once` |
| Guessing secrets / hammering | rate limits (browser: per IP and site+user; server: per IP and site) | `test_siteverify_brute_force_is_rate_limited`, `test_rate_limit_applies_to_browser_endpoints` |
| Missing signing key | refuses to start | `test_refuses_to_start_without_a_proper_key` |

## Not covered
- A stolen valid token used within its 60 s (it is bound to the site and user, not to the visitor's connection).
- A customer server that skips `siteverify`, ignores `rhythm_match`, or does not compare the returned `user_id` with
  the account it is logging in.
- Replay of captured raw timings is handled by the engine's replay check (it shows up as `rhythm_match=false`), not
  by this layer. Timings forged to imitate a victim are the biometric's own error rate (see RESULTS.md).
- Lockout can be triggered on purpose against a known id by anyone who can reach `/v1/verify` from an allowed origin
  (same as the single-tenant API); the per-site+user rate limit slows it only.
- The Origin header is a browser rule, not authentication: scripts outside a browser can set it. The ticket and the
  secret are what protect enrollment and decisions.
- Timing differences between outcomes (an unknown user returns faster than a scored attempt) are not equalized.
- Rate limits and the jti table: limits are per process; the jti table is shared through SQLite, so several workers
  on one database file are fine, several hosts are not. No key rotation scheme with overlap, no per-site rate quotas,
  no audit log, no admin HTTP API (CLI only).
