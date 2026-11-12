# demo_site: an independent "customer" site that embeds keyauth

A small web shop login (register, log in, account) that plays a third-party customer of the keyauth service
(`docs/SERVICE.md`). It is deliberately **not** part of `src/keyauth`: it has its own SQLite file, its own users,
its own passwords and sessions, and it talks to keyauth only over HTTP, the way any customer would. FastAPI, no other dependency.

```
python -m demo_site          # run from the repository root; configuration is environment variables only
```

## Configuration (copy into a local `.env` or export; nothing here is committed)
```
# .env.example style. Required:
KEYAUTH_URL=http://127.0.0.1:8765          # where keyauth runs (browser and server use the same URL here)
KEYAUTH_SITE_KEY=pk_...                    # PUBLIC site key (goes into the pages)
KEYAUTH_SITE_SECRET=sk.<site_id>.<random>  # SECRET: server side only, never rendered, never logged
# Optional:
DEMO_SITE_DB=demo_site.db                  # own SQLite file (gitignored: *.db)
DEMO_SITE_HOST=127.0.0.1
DEMO_SITE_PORT=9000
DEMO_SITE_FAIL_MODE=open                   # open | closed: what "no rhythm check available" means, see Policy
DEMO_SITE_SECURE_COOKIE=0                  # 1 behind https: adds the Secure flag to the session cookie
DEMO_SITE_PEPPER=                          # key for stand-in ids of unknown usernames; default derived from the secret
```
Register the site's origin once, on the keyauth side (prints the site key and secret once):
```
export KEYAUTH_ENABLE_V1=1 KEYAUTH_TOKEN_KEY=$(python -c "import secrets; print(secrets.token_urlsafe(48))")
python -m keyauth.sites create --name "Demo Shop" --origin http://127.0.0.1:9000
python -m keyauth.api        # keyauth on :8765
```
`scripts/run_integration_e2e.sh` does all of this on throw-away databases and runs the browser test
`tests/e2e/integration.e2e.mjs`. Server logic is covered by `tests/test_demo_site.py` (a stub keyauth, no model).

## Flow
1. **Register**: the site stores the user (scrypt, random salt, constant-time compare) and a random opaque id
   `u_<32 hex>`. The id is never derived from the username or email; it is the only user identifier keyauth ever sees.
   Registration logs the user in.
2. **Enroll** (account page, logged-in session only): the button calls the site's `POST /account/ticket`. The site
   takes the opaque id from the **server-side session** (not from the request), calls `POST /v1/tickets` with its
   secret, and returns `{ticket, user_id, sitekey}`. The page then runs `keyauth.enroll` 10 times with that ticket.
   No ticket is ever issued at registration time before the password was verified, to a pending (second-step) session,
   or to an anonymous visitor.
3. **Login**: `login.js` (same-origin file) asks `POST /login/identify` for the opaque id when the username field
   changes, stores it in a hidden input **without a `name`** (so it is never posted) and calls
   `keyauth.attach(password, {sitekey, userIdField: "#keyauth-uid"})`. On `POST /login` the server:
   1. looks the account up and verifies the password hash (scrypt runs for unknown names too). Wrong -> one generic
      error, **keyauth is not called** and the token stays unspent;
   2. only then calls `POST /v1/siteverify` with the form's `keyauth_token`;
   3. applies the policy below, comparing the token's `user_id` with **this account's** opaque id.

### Why identify does not leak whether a username exists
The browser must tell keyauth *some* user id before the password is typed, so the page needs an id before the site has
authenticated anyone. `/login/identify` answers every username with the same shape: the real opaque id for an
existing account, and for an unknown name a **stand-in** `u_` + 32 hex derived as `HMAC(pepper, username)`. Both look
the same, the stand-in is stable per name, and keyauth answers `/v1/verify` identically for unknown users (always a
token). The attacker learns nothing from the identify step. What it deliberately does not hide: **registration**
says "username taken" (it has to), and a rate limiter on `/login/identify` and `/login` is not implemented in this demo.
A `/whoami`-style step that returns 404 for unknown names was rejected: it is an enumeration oracle.

### Policy (the customer's choice, not keyauth's; implemented in `decide_login`)
| Password | Token / keyauth | Result |
|---|---|---|
| wrong or unknown user | any | rejected, generic message, keyauth not called |
| ok | `success` and `user_id` == account and `rhythm_match` | **logged in** (log: risk) |
| ok | `success` and `rhythm_match` false | **second step** (not a hard deny) |
| ok | `success` but `user_id` is another account's | second step |
| ok | `success: false` (tampered, expired, reused, wrong site) or oversized token | second step |
| ok | empty token, or keyauth unreachable / answers non-200 | **fail-open to password only**, logged as `FLAG ...`; session shows "password only" |

The second step is a stub: a six digit code is printed to the server log (`SECOND-STEP CODE`) instead of an email;
five tries, then the pending session is dropped. A pending session is not a login (no account page, no ticket).
`DEMO_SITE_FAIL_MODE=closed` makes the two fail-open rows a second step instead. Know the trade-off: with fail-open, anyone
who has the password can simply omit the token, so the rhythm check only protects users whose browser ran
`keyauth.js`. That is the price of never locking users out when keyauth is down; a real shop picks its own side.

## Strict CSP
Every response carries
`default-src 'none'; script-src 'self' <KEYAUTH_URL>; connect-src 'self' <KEYAUTH_URL>; style-src 'self'; img-src 'self'; form-action 'self'; base-uri 'none'; frame-ancestors 'none'`.
There is no inline script or style and no inline event handler. The pages carry only the public site key.

## Security notes
- Session cookie `demo_session`: random 256-bit id, only its SHA-256 is stored server-side; `HttpOnly; SameSite=Lax; Path=/`
  (`Secure` with `DEMO_SITE_SECURE_COOKIE=1`); a fresh id on every login (no fixation); logout deletes the row.
- **CSRF**: no CSRF tokens. `SameSite=Lax` keeps the cookie off cross-site POSTs, and the JSON endpoints need
  `application/json`. A production site should still add tokens (or check `Origin`). Documented, not implemented.
- Not implemented: login rate limiting / account lockout, password reset, email verification, HTTPS termination.
- Server-to-server calls send no `Origin` header (keyauth refuses those) and ignore proxy environment variables.
- Enrollment types the password again 10 times; the site cannot verify that the typed text is the password (it never
  sees keystrokes), keyauth only checks that all samples have the same length.
