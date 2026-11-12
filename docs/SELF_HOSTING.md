# Self-hosting Tempo

Run the `/v1` service on your own machine and use it from your own website, in about 10 minutes. You need Docker
(with Compose v2), a website with a server you control, and a domain with TLS for production.

Read [What you are responsible for](#what-you-are-responsible-for) and [What this is, and is not](#what-this-is-and-is-not) before you ship it.

## Quickstart

```bash
git clone <this repository> tempo && cd tempo
sh scripts/selfhost_init.sh
```

`selfhost_init.sh` creates `.env` (mode 600, gitignored) with a random `KEYAUTH_TOKEN_KEY` (signs tokens) and
`KEYAUTH_ADMIN_TOKEN`, and your user/group id for the training job. It never overwrites a value that is already
there and never prints a secret. Compose refuses to start without `KEYAUTH_TOKEN_KEY`.

**Get a model.** The image does not contain one. Train on the CMU benchmark (about 2 minutes on a laptop CPU; needs
network to download the data from cmu.edu):

```bash
docker compose --profile train run --rm train
```

This builds the image, downloads the data into `./data` and writes `./checkpoints/model.pt` (both gitignored). Or put
a `model.pt` you trained elsewhere into `./checkpoints/`. Without a model the service starts, `/health` says
`"model_loaded": false`, and enroll/verify answer 503.

**Start it.**

```bash
docker compose up -d --build
curl http://127.0.0.1:8000/health        # {"status":"ok","model_loaded":true}
```

**Create a site** (one per website; the origin is the exact scheme + host of your login page):

```bash
docker compose exec tempo python -m keyauth.sites create --name "My shop" --origin https://shop.example
```

It prints the public **site key** (`pk_...`, goes in your page) and the **secret** (`sk....`, server side only).
The secret is shown once and only a hash is stored: put it in your server's secret store now. Other commands:
`list`, `disable SITE_ID`, `enable SITE_ID`, `rotate-secret SITE_ID`. Add more origins by repeating `--origin`
(`http://localhost:PORT` is allowed for testing; other `http` origins are not).

**Add it to your login page.** The page loads the script from your Tempo host; use your public HTTPS URL
(see [TLS](#tls-reverse-proxy)):

```html
<script src="https://tempo.example.com/v1/keyauth.js" defer></script>
<script src="/login-page.js" defer></script>
```
```js
// login-page.js (same-origin file)
keyauth.attach(document.getElementById("pw"), {sitekey: "pk_...", userIdField: "#keyauth-uid"});
```

`userIdField` holds an **opaque id** for the account (`[A-Za-z0-9._-]{1,64}`, not an email or username). The form now
posts an extra field `keyauth_token`. If your page has a CSP, `script-src` and `connect-src` must include the Tempo
host. Enrollment (`keyauth.enroll`, 10 typed samples) is described in [SERVICE.md](SERVICE.md#embedding).

**Add it to your server.** Copy one file, `integrations/python/tempo_verify.py` or `integrations/node/tempo-verify.mjs`
(standard library only), and after you checked the password:

```python
from tempo_verify import verify, issue_ticket
r = verify(os.environ["TEMPO_SECRET"], request.form.get("keyauth_token", ""), "https://tempo.example.com")
ok = r.success and r.user_id == account.tempo_id and r.rhythm_match      # else: second factor, or fail open
# to enroll a logged-in user, hand the page a ticket:  issue_ticket(secret, account.tempo_id, url).ticket
```
```js
import { verify, issueTicket } from "./tempo-verify.mjs";
const r = await verify(process.env.TEMPO_SECRET, req.body.keyauth_token, "https://tempo.example.com");
const ok = r.success && r.user_id === account.tempoId && r.rhythm_match;   // else: second factor, or fail open
```

Both return `success=False` with a `reason` (`unreachable`, `unauthorized`, `invalid_token`, `rate_limited`,
`bad_response`, `no_token`) instead of raising, so you decide what happens when Tempo is down or the token is
missing. Fail-open (password alone) never locks users out; fail-closed (ask for a second factor) is stricter. A token
is single use and lives 60 seconds; always compare `user_id` with the account that just logged in. The secret never
goes in a page, a log, or a URL.

**Try it with the demo shop.** [`demo_site/`](../demo_site/README.md) is a complete customer site. Create a site for
`http://127.0.0.1:9000`, then run it with `KEYAUTH_URL=http://127.0.0.1:8000`, `KEYAUTH_SITE_KEY` and
`KEYAUTH_SITE_SECRET` from the `create` output.

## Settings

Put them in `.env` (compose reads it) or export them. Only these reach the container; the others in
[.env.example](../.env.example) are fixed by the image or compose file.

| Variable | Default | Meaning |
|---|---|---|
| `KEYAUTH_TOKEN_KEY` | none, required | HMAC key that signs tokens, 32+ characters. Rotating it invalidates tokens in flight (at most 60 s). |
| `KEYAUTH_ADMIN_TOKEN` | empty | Bearer token for `DELETE /user` (single-tenant API only, see [Deleting data](#backups-and-deleting-data)). 16+ characters. |
| `KEYAUTH_SERVER_RATE_LIMIT` | `600/60s` | `/v1/siteverify` and `/v1/tickets`, per client IP and per site, per process. |
| `KEYAUTH_RATE_LIMIT` | `30/60s` | Browser endpoints, per client IP and per user id, per process. |
| `KEYAUTH_PER_USER` | `1` | Per-user thresholds from the model's calibration. |
| `KEYAUTH_SERVE_UI` | `0` | `1` serves the single-tenant demo page at `/`. Not needed for `/v1`. |
| `TEMPO_BIND` / `TEMPO_PORT` | `127.0.0.1` / `8000` | Host address and port of the published port. |
| `TEMPO_UID` / `TEMPO_GID` | your ids | User of the one-off `train` job, so `./checkpoints` is yours. |

Set by the image: `KEYAUTH_ENABLE_V1=1` (compose), `KEYAUTH_CHECKPOINT=/app/checkpoints/model.pt`,
`KEYAUTH_DB=/app/state/keyauth.db`, `KEYAUTH_HOST=0.0.0.0`, `KEYAUTH_PORT=8000`. The container runs as a non-root user.

## TLS reverse proxy

The port is published on `127.0.0.1` only. Browsers on your login page need HTTPS (an https page cannot load an http
script), and the secret and tokens must not cross the network in clear text, so put a TLS proxy in front. The
compose file does not run one. Example `Caddyfile` on the same host (Caddy gets and renews the certificate):

```
tempo.example.com {
    encode zstd gzip
    reverse_proxy 127.0.0.1:8000
}
```

Your own server can reach Tempo through that URL, or directly at `http://127.0.0.1:8000` if it runs on the same host
(plain HTTP over loopback only). Do not set `TEMPO_BIND=0.0.0.0` without a firewall or proxy in front. Behind a proxy
the per-IP rate limits see the proxy's address; the per-site and per-user limits still apply. (uvicorn can trust
`X-Forwarded-For` via its `FORWARDED_ALLOW_IPS` environment variable; this is not set up or tested here.)

## Backups and deleting data

All state is one SQLite file in the `tempo-state` Docker volume (`/app/state/keyauth.db`): sites (salted secret
hashes), enrolled **templates**, enrollment/verification embeddings, and used-token ids. It contains **biometric
data**: derived from typing rhythm, not raw keystrokes (the server never sees what was typed), but still a
biometric template of a person.

- **Encrypt the disk** that holds the volume and the backups. Restrict who can read the Docker host.
- **Back up** with the service stopped, or with SQLite's online backup, never by copying a file under write:
  ```bash
  docker compose stop tempo
  docker run --rm -v tempo_tempo-state:/s -v "$PWD":/b alpine tar czf /b/tempo-state.tgz -C /s .
  docker compose start tempo
  ```
  (The volume is named `<project>_tempo-state`; `docker volume ls` shows it.) Treat the archive as biometric data too, and
  expire old backups: a deleted user lives on in every backup until it is rotated out. Restoring is the reverse
  `tar xzf` into the volume. Without a backup, a lost volume means every user must enroll again.
- **Retention:** decide how long you keep the templates of users who left. There is no automatic expiry.
- **Deleting one user:** the customer's server calls `issue_ticket(secret, user_id, url, reset=True)`. Reset deletes
  the user's template and samples immediately; the new ticket it returns can simply be ignored (it expires after 10
  minutes). `DELETE /user` with `KEYAUTH_ADMIN_TOKEN` exists too, but it only reaches users of the single-tenant API
  (`app_id` `default`), not users created through `/v1` sites. Disabling a site (`sites disable`) stops it from
  verifying but does not delete its users' data; to erase everything, delete the volume (`docker compose down -v`).

## Upgrades

```bash
git pull
docker compose up -d --build
```

Back up first. The database lives in the volume and survives the rebuild; the app adds missing tables or columns
when it starts, but there is no migration tool or downgrade path. A model trained by an older version may need
retraining; read the changes before upgrading. Rotating a site secret (`sites rotate-secret`) invalidates the old one at once.

## Limits you must know

- **One process, one replica.** State is a local SQLite file and rate limits live in process memory. Do not run several
  replicas or put the file on a network share. Several hosts are not supported.
- **No admin API or UI** beyond the CLI above, no audit log, no per-site quotas.
- **Lockout can be provoked:** anyone who can reach `/v1/verify` from an allowed origin and knows a user id can trigger
  the lockout. The customer then sees `rhythm_match=false`.
- Threat model and what is not covered: [SERVICE.md](SERVICE.md#threat-model-what-each-failure-test-defends).

## What you are responsible for

You run the service, so you are the operator of a biometric system. Keystroke rhythm used to recognize a person is
biometric data in many places (for example BIPA in Illinois, GDPR Article 9 in the EU and UK, and other state and
national laws). This is not legal advice; ask someone qualified. At minimum:

- **Consent:** tell users plainly what you collect and why, and get explicit, recorded consent before they enroll. Offer an
  alternative that does not need it.
- **Retention:** keep templates only as long as needed, publish the period, and delete on schedule and when the user
  leaves or withdraws consent (including from backups).
- **Deletion and access:** be able to delete a user's data on request (see above).
- **Security:** TLS, encrypted disks, restricted access to the host, secrets in a secret store, and a plan for a breach.
- Do not share or sell templates, and do not use them for anything but this login check.

## What this is, and is not

Tempo is a **soft signal**. On the CMU benchmark the model has about **21% equal error rate** on unseen users (see the
[README](../README.md#results)): at that setting roughly one in five genuine logins looks wrong and a similar share of
impostors passes. Use it to add friction (a second factor, an email, extra monitoring) when the rhythm does not match,
not to block logins or as a replacement for the password. It is a demo-grade service, not a hardened product.
