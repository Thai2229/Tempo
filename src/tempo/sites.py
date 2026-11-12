"""Site management for the multi-site service, and the admin CLI.

    python -m tempo.sites create --name "My shop" --origin https://shop.example [--origin ...]
    python -m tempo.sites list
    python -m tempo.sites disable SITE_ID        (enable: undo)
    python -m tempo.sites rotate-secret SITE_ID

`create` and `rotate-secret` print the public site key and the secret ONCE. Only a salted SHA-256 of the secret is
stored (the secret is 256 random bits, so a fast hash is enough and a slow one would only make /v1/siteverify
costly), so it can never be shown again. The database is TEMPO_DB unless --db is given.

Secret format: "sk.<site_id>.<random>". The site id is not secret; it lets /v1/siteverify find the site from the
secret alone.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import os
import secrets
import sys
import time
from urllib.parse import urlsplit

from tempo.auth.store import SiteRecord, TemplateStore


def hash_secret(salt: str, secret: str) -> str:
    return hashlib.sha256(bytes.fromhex(salt) + secret.encode()).hexdigest()


def normalize_origin(origin: str) -> str:
    """'https://Example.com' -> 'https://example.com'. Scheme + host[:port] only; https unless localhost."""
    parts = urlsplit(origin.strip())
    host = parts.hostname
    if parts.scheme not in {"http", "https"} or not host or parts.path not in {"", "/"} or parts.query or parts.fragment or parts.username or parts.password:
        raise ValueError(f"origin must look like https://example.com (no path), got {origin!r}")
    if parts.scheme == "http" and host not in {"localhost", "127.0.0.1"}:
        raise ValueError("http origins are only allowed for localhost / 127.0.0.1")
    result = f"{parts.scheme}://{host.lower()}"
    if parts.port is not None:
        result += f":{parts.port}"
    return result


def secret_site_id(secret: object) -> str | None:
    """The site id claimed by a secret string (not yet authenticated), or None if malformed."""
    if not isinstance(secret, str) or len(secret) > 256:
        return None
    parts = secret.split(".")
    if len(parts) != 3 or parts[0] != "sk" or not parts[1] or not parts[2]:
        return None
    return parts[1]


def authenticate_secret(store: TemplateStore, secret: object) -> SiteRecord | None:
    """The enabled site this secret belongs to, else None. Constant-time compare; same work for unknown sites."""
    site_id = secret_site_id(secret)
    site = None
    if site_id is not None:
        site = store.get_site(site_id)
    salt = "00" * 16
    expected = "0" * 64
    if site is not None:
        salt, expected = site.secret_salt, site.secret_hash
    text = ""
    if isinstance(secret, str):
        text = secret
    given = hash_secret(salt, text)
    ok = hmac.compare_digest(given.encode(), expected.encode())
    if ok and site is not None and not site.disabled:
        return site
    return None


def _new_secret(site_id: str) -> tuple[str, str, str]:
    secret = f"sk.{site_id}.{secrets.token_urlsafe(32)}"
    salt = secrets.token_hex(16)
    return secret, salt, hash_secret(salt, secret)


def create_site(store: TemplateStore, name: str, origins: list[str], now: float | None = None) -> tuple[SiteRecord, str]:
    """Create a site. Returns (record, secret); the secret exists nowhere else."""
    if not name.strip() or not origins:
        raise ValueError("a site needs a name and at least one origin")
    if now is None:
        now = time.time()
    site_id = "s" + secrets.token_hex(6)
    secret, salt, secret_hash = _new_secret(site_id)
    site = SiteRecord(
        site_id=site_id,
        name=name.strip(),
        site_key="pk_" + secrets.token_urlsafe(18),
        secret_salt=salt,
        secret_hash=secret_hash,
        origins=sorted({normalize_origin(o) for o in origins}),
        created_at=now,
        disabled=False,
    )
    store.create_site(site)
    return site, secret


def rotate_secret(store: TemplateStore, site_id: str) -> str | None:
    secret, salt, secret_hash = _new_secret(site_id)
    if not store.set_site_secret(site_id, salt, secret_hash):
        return None
    return secret


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m tempo.sites", description=__doc__.split("\n")[0])
    parser.add_argument("--db", default=os.environ.get("TEMPO_DB", "tempo.db"))
    sub = parser.add_subparsers(dest="cmd", required=True)
    create = sub.add_parser("create")
    create.add_argument("--name", required=True)
    create.add_argument("--origin", action="append", required=True, dest="origins")
    sub.add_parser("list")
    for name in ("disable", "enable", "rotate-secret"):
        sub.add_parser(name).add_argument("site_id")
    args = parser.parse_args(argv)

    store = TemplateStore(args.db)
    try:
        if args.cmd == "create":
            try:
                site, secret = create_site(store, args.name, args.origins)
            except ValueError as e:
                print(f"error: {e}", file=sys.stderr)
                return 2
            print(f"site_id:  {site.site_id}")
            print(f"site key: {site.site_key}   (public, goes in the page)")
            print(f"secret:   {secret}")
            print("Store the secret now in the customer's server config. It is not stored and cannot be shown again.")
            return 0
        if args.cmd == "list":
            for s in store.list_sites():
                state = "disabled"
                if not s.disabled:
                    state = "active"
                print(f"{s.site_id}  {state:8}  {s.site_key}  {s.name}  {' '.join(s.origins)}")
            return 0
        if args.cmd in {"disable", "enable"}:
            if not store.set_site_disabled(args.site_id, args.cmd == "disable"):
                print("error: unknown site", file=sys.stderr)
                return 1
            print(f"{args.cmd}d {args.site_id}")
            return 0
        secret = rotate_secret(store, args.site_id)
        if secret is None:
            print("error: unknown site", file=sys.stderr)
            return 1
        print(f"new secret: {secret}")
        print("The old secret no longer works. This one is not stored and cannot be shown again.")
        return 0
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
