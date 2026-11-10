"""SQLite storage for user templates and recent embeddings.

What is stored, per (app_id, user_id):
  users    template embedding, password length, status, lockout state
  samples  embeddings of enrollment attempts and recent accepted logins
           (used to rebuild templates and to detect replayed attempts); in
           scorer mode the engine stores raw timing vectors in seconds instead

Raw timings and typed characters are never stored. app_id namespaces
everything, so two apps can both have a user called "alice".
"""

from __future__ import annotations

import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    app_id        TEXT NOT NULL,
    user_id       TEXT NOT NULL,
    n_keys        INTEGER NOT NULL,
    status        TEXT NOT NULL,           -- 'enrolling' | 'active'
    template      BLOB,                    -- float32 unit vector, NULL while enrolling / in scorer mode
    threshold     REAL,                    -- per-user override, NULL = engine default
    scorer_state  TEXT,                    -- JSON ScorerState (scorer mode only), NULL otherwise
    failures      INTEGER NOT NULL DEFAULT 0,
    locked_until  REAL NOT NULL DEFAULT 0,
    created_at    REAL NOT NULL,
    updated_at    REAL NOT NULL,
    PRIMARY KEY (app_id, user_id)
);
CREATE TABLE IF NOT EXISTS samples (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    app_id      TEXT NOT NULL,
    user_id     TEXT NOT NULL,
    kind        TEXT NOT NULL,             -- 'enroll' | 'verify'
    embedding   BLOB NOT NULL,
    created_at  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS samples_user ON samples (app_id, user_id, kind, id);
"""


@dataclass
class UserRecord:
    app_id: str
    user_id: str
    n_keys: int
    status: str
    template: np.ndarray | None
    threshold: float | None
    failures: int
    locked_until: float
    scorer_state: str | None = None


def _to_blob(v: np.ndarray) -> bytes:
    return np.asarray(v, dtype=np.float32).tobytes()


def _from_blob(b: bytes) -> np.ndarray:
    return np.frombuffer(b, dtype=np.float32).copy()


class TemplateStore:
    def __init__(self, path: str | Path = ":memory:"):
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock, self._conn:
            self._conn.executescript(_SCHEMA)
            # databases created before scorer mode lack the column
            columns = [r[1] for r in self._conn.execute("PRAGMA table_info(users)").fetchall()]
            if "scorer_state" not in columns:
                self._conn.execute("ALTER TABLE users ADD COLUMN scorer_state TEXT")

    def close(self) -> None:
        self._conn.close()

    # ---- users ----------------------------------------------------------

    def get_user(self, app_id: str, user_id: str) -> UserRecord | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT n_keys, status, template, threshold, failures, locked_until, scorer_state "
                "FROM users WHERE app_id = ? AND user_id = ?",
                (app_id, user_id),
            ).fetchone()
        if row is None:
            return None
        template = None
        if row[2] is not None:
            template = _from_blob(row[2])
        return UserRecord(app_id, user_id, row[0], row[1], template, row[3], row[4], row[5], row[6])

    def create_user(self, app_id: str, user_id: str, n_keys: int, now: float) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO users (app_id, user_id, n_keys, status, created_at, updated_at) "
                "VALUES (?, ?, ?, 'enrolling', ?, ?)",
                (app_id, user_id, n_keys, now, now),
            )

    def set_template(self, app_id: str, user_id: str, template: np.ndarray, now: float) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE users SET template = ?, status = 'active', updated_at = ? WHERE app_id = ? AND user_id = ?",
                (_to_blob(template), now, app_id, user_id),
            )

    def set_scorer_state(self, app_id: str, user_id: str, state_json: str, now: float) -> None:
        """Scorer-mode counterpart of set_template: keeps the fitted state and activates the user."""
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE users SET scorer_state = ?, status = 'active', updated_at = ? WHERE app_id = ? AND user_id = ?",
                (state_json, now, app_id, user_id),
            )

    def set_threshold(self, app_id: str, user_id: str, threshold: float | None) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE users SET threshold = ? WHERE app_id = ? AND user_id = ?", (threshold, app_id, user_id)
            )

    def set_lockout(self, app_id: str, user_id: str, failures: int, locked_until: float) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE users SET failures = ?, locked_until = ? WHERE app_id = ? AND user_id = ?",
                (failures, locked_until, app_id, user_id),
            )

    def delete_user(self, app_id: str, user_id: str) -> bool:
        with self._lock, self._conn:
            cur = self._conn.execute("DELETE FROM users WHERE app_id = ? AND user_id = ?", (app_id, user_id))
            self._conn.execute("DELETE FROM samples WHERE app_id = ? AND user_id = ?", (app_id, user_id))
        return cur.rowcount > 0

    # ---- samples --------------------------------------------------------

    def add_sample(self, app_id: str, user_id: str, kind: str, embedding: np.ndarray, now: float) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO samples (app_id, user_id, kind, embedding, created_at) VALUES (?, ?, ?, ?, ?)",
                (app_id, user_id, kind, _to_blob(embedding), now),
            )

    def get_samples(self, app_id: str, user_id: str, kind: str | None = None) -> np.ndarray:
        """(k, D) embeddings, oldest first. Empty (0, 0) array if none."""
        sql = "SELECT embedding FROM samples WHERE app_id = ? AND user_id = ?"
        params: tuple = (app_id, user_id)
        if kind is not None:
            sql += " AND kind = ?"
            params += (kind,)
        with self._lock:
            rows = self._conn.execute(sql + " ORDER BY id", params).fetchall()
        if not rows:
            return np.zeros((0, 0), dtype=np.float32)
        return np.stack([_from_blob(r[0]) for r in rows])

    def count_samples(self, app_id: str, user_id: str, kind: str) -> int:
        with self._lock:
            return self._conn.execute(
                "SELECT COUNT(*) FROM samples WHERE app_id = ? AND user_id = ? AND kind = ?", (app_id, user_id, kind)
            ).fetchone()[0]

    def clear_samples(self, app_id: str, user_id: str, kind: str) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "DELETE FROM samples WHERE app_id = ? AND user_id = ? AND kind = ?", (app_id, user_id, kind)
            )

    def trim_samples(self, app_id: str, user_id: str, kind: str, keep: int) -> None:
        """Keep only the newest `keep` samples of this kind."""
        with self._lock, self._conn:
            self._conn.execute(
                "DELETE FROM samples WHERE app_id = ? AND user_id = ? AND kind = ? AND id NOT IN ("
                "  SELECT id FROM samples WHERE app_id = ? AND user_id = ? AND kind = ? ORDER BY id DESC LIMIT ?)",
                (app_id, user_id, kind, app_id, user_id, kind, keep),
            )
