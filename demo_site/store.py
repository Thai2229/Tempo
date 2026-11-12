"""The site's own SQLite file: users (scrypt password hashes, opaque ids) and server-side sessions."""

from __future__ import annotations

import base64
import hashlib
import hmac
import secrets
import sqlite3
import time
from dataclasses import dataclass
from typing import Callable

SCRYPT_N = 2**14
SCRYPT_R = 8
SCRYPT_P = 1
SCRYPT_MAXMEM = 64 * 1024 * 1024

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY,
    username TEXT NOT NULL UNIQUE COLLATE NOCASE,
    opaque_id TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    created REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS sessions (
    sid_hash TEXT PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id),
    state TEXT NOT NULL CHECK (state IN ('pending', 'active')),
    code_hash TEXT,
    attempts INTEGER NOT NULL DEFAULT 0,
    method TEXT NOT NULL DEFAULT '',
    expires REAL NOT NULL
);
"""


def hash_password(password: str, salt: bytes | None = None) -> str:
    """'scrypt$N$r$p$salt$hash' (salt and hash base64). A fresh random salt per password."""
    if salt is None:
        salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode(), salt=salt, n=SCRYPT_N, r=SCRYPT_R, p=SCRYPT_P, maxmem=SCRYPT_MAXMEM, dklen=32)
    return "$".join(["scrypt", str(SCRYPT_N), str(SCRYPT_R), str(SCRYPT_P), base64.b64encode(salt).decode(), base64.b64encode(digest).decode()])


def check_password(password: str, stored: str) -> bool:
    try:
        scheme, n, r, p, salt_b64, hash_b64 = stored.split("$")
        salt = base64.b64decode(salt_b64)
        expected = base64.b64decode(hash_b64)
        if scheme != "scrypt":
            return False
        digest = hashlib.scrypt(password.encode(), salt=salt, n=int(n), r=int(r), p=int(p), maxmem=SCRYPT_MAXMEM, dklen=len(expected))
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(digest, expected)


def sid_hash(sid: str) -> str:
    """Only a hash of the session id is stored, so a copy of the database cannot be replayed as cookies."""
    return hashlib.sha256(sid.encode()).hexdigest()


@dataclass(frozen=True)
class User:
    id: int
    username: str
    opaque_id: str
    password_hash: str


@dataclass(frozen=True)
class Session:
    user: User
    state: str
    code_hash: str | None
    attempts: int
    method: str  # how this session was authenticated: rhythm, password_only, code (shown on the account page)


class Store:
    def __init__(self, path: str, clock: Callable[[], float] = time.time) -> None:
        self.path = path
        self.clock = clock
        with self._connect() as db:
            db.executescript(SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path, timeout=10)
        db.execute("PRAGMA foreign_keys = ON")
        return db

    def create_user(self, username: str, password: str) -> User | None:
        """None if the username is taken. The opaque id is random and unrelated to the username."""
        opaque_id = "u_" + secrets.token_hex(16)
        stored = hash_password(password)
        try:
            with self._connect() as db:
                cur = db.execute(
                    "INSERT INTO users (username, opaque_id, password_hash, created) VALUES (?, ?, ?, ?)",
                    (username, opaque_id, stored, self.clock()),
                )
        except sqlite3.IntegrityError:
            return None
        return User(int(cur.lastrowid), username, opaque_id, stored)

    def get_user(self, username: str) -> User | None:
        with self._connect() as db:
            row = db.execute("SELECT id, username, opaque_id, password_hash FROM users WHERE username = ?", (username,)).fetchone()
        if row is None:
            return None
        return User(*row)

    def create_session(self, user: User, state: str, ttl_s: float, code_hash: str | None = None, method: str = "") -> str:
        sid = secrets.token_urlsafe(32)
        with self._connect() as db:
            db.execute("DELETE FROM sessions WHERE expires < ?", (self.clock(),))
            db.execute(
                "INSERT INTO sessions (sid_hash, user_id, state, code_hash, attempts, method, expires) VALUES (?, ?, ?, ?, 0, ?, ?)",
                (sid_hash(sid), user.id, state, code_hash, method, self.clock() + ttl_s),
            )
        return sid

    def get_session(self, sid: str | None) -> Session | None:
        if not sid:
            return None
        with self._connect() as db:
            row = db.execute(
                "SELECT u.id, u.username, u.opaque_id, u.password_hash, s.state, s.code_hash, s.attempts, s.method "
                "FROM sessions s JOIN users u ON u.id = s.user_id WHERE s.sid_hash = ? AND s.expires >= ?",
                (sid_hash(sid), self.clock()),
            ).fetchone()
        if row is None:
            return None
        return Session(User(*row[:4]), row[4], row[5], row[6], row[7])

    def delete_session(self, sid: str | None) -> None:
        if sid:
            with self._connect() as db:
                db.execute("DELETE FROM sessions WHERE sid_hash = ?", (sid_hash(sid),))

    def count_attempt(self, sid: str) -> None:
        with self._connect() as db:
            db.execute("UPDATE sessions SET attempts = attempts + 1 WHERE sid_hash = ?", (sid_hash(sid),))
