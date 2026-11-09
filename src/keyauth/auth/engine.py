"""Enrollment and verification.

The engine works on timing arrays only. It never sees the password: the
integrating app checks the password itself and asks keyauth whether the
RHYTHM matches.

Enrollment: the user types their password `enroll_samples` times. Each
attempt's embedding is stored; once enough are collected the template is
their normalized mean. If the user's own attempts disagree with each other
too much (median distance to the template above the threshold), enrollment
restarts, because those attempts would not pass verification either.

Verification: distance from the attempt's embedding to the template, accept
if <= threshold. Also:
  - length check: the attempt must have the same number of keys as enrollment
  - replay check: an embedding (near-)identical to a stored one is rejected;
    humans never type the same timings twice, copied requests do
  - lockout: after max_failures consecutive rejections, refuse for lockout_s
  - adaptation: confident accepts nudge the template toward the user's
    current typing, since rhythm drifts over weeks
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np

from keyauth.auth.store import TemplateStore
from keyauth.auth.validation import validate_timings
from keyauth.evaluate import make_template


class Embedder(Protocol):
    def __call__(self, timings: np.ndarray) -> np.ndarray:
        """(n, 3) validated timings -> unit-length embedding."""


@dataclass
class EngineConfig:
    enroll_samples: int = 10
    replay_epsilon: float = 1e-4  # embedding distance below which an attempt is a replay
    history_size: int = 50  # accepted-login embeddings kept for replay checks
    max_failures: int = 5
    lockout_s: float = 300.0
    adapt: bool = True
    adapt_alpha: float = 0.1  # weight of a new attempt in the template update
    adapt_max_ratio: float = 0.5  # only adapt when distance <= ratio * threshold


@dataclass
class EnrollResult:
    status: str  # "in_progress" | "complete" | "rejected"
    samples: int
    required: int
    reason: str | None = None  # set when rejected


@dataclass
class VerifyResult:
    accepted: bool
    reason: str  # "match" | "no_match" | "not_enrolled" | "locked" | "length_mismatch" | "replay"
    score: float | None  # distance to template, lower = more similar
    threshold: float


class KeyAuthEngine:
    def __init__(
        self,
        embedder: Embedder,
        store: TemplateStore,
        threshold: float,
        config: EngineConfig | None = None,
        clock: Callable[[], float] = time.time,
    ):
        if config is None:
            config = EngineConfig()
        self.embedder = embedder
        self.store = store
        self.threshold = float(threshold)
        self.config = config
        self.clock = clock

    # ---- enrollment -----------------------------------------------------

    def enroll(self, app_id: str, user_id: str, timings) -> EnrollResult:
        required = self.config.enroll_samples
        t = validate_timings(timings)
        now = self.clock()

        user = self.store.get_user(app_id, user_id)
        if user is None:
            self.store.create_user(app_id, user_id, n_keys=len(t), now=now)
        elif user.status == "active":
            return EnrollResult("rejected", required, required, "already_enrolled")
        elif user.n_keys != len(t):
            count = self.store.count_samples(app_id, user_id, "enroll")
            return EnrollResult("rejected", count, required, "length_mismatch")

        emb = self._embed(t)
        if self._is_replay(app_id, user_id, emb):
            count = self.store.count_samples(app_id, user_id, "enroll")
            return EnrollResult("rejected", count, required, "replay")

        self.store.add_sample(app_id, user_id, "enroll", emb, now)
        count = self.store.count_samples(app_id, user_id, "enroll")
        if count < required:
            return EnrollResult("in_progress", count, required)

        samples = self.store.get_samples(app_id, user_id, "enroll")
        template = make_template(samples)
        spread = float(np.median(np.linalg.norm(samples - template, axis=1)))
        if spread > self._threshold_for(user):
            self.store.clear_samples(app_id, user_id, "enroll")
            return EnrollResult("rejected", 0, required, "inconsistent")

        self.store.set_template(app_id, user_id, template, now)
        return EnrollResult("complete", count, required)

    # ---- verification ---------------------------------------------------

    def verify(self, app_id: str, user_id: str, timings) -> VerifyResult:
        user = self.store.get_user(app_id, user_id)
        threshold = self._threshold_for(user)
        if user is None or user.status != "active":
            return VerifyResult(False, "not_enrolled", None, threshold)

        now = self.clock()
        if user.locked_until > now:
            return VerifyResult(False, "locked", None, threshold)

        t = validate_timings(timings)
        if len(t) != user.n_keys:
            self._record_failure(user, now)
            return VerifyResult(False, "length_mismatch", None, threshold)

        emb = self._embed(t)
        if self._is_replay(app_id, user_id, emb):
            self._record_failure(user, now)
            return VerifyResult(False, "replay", None, threshold)

        score = float(np.linalg.norm(emb - user.template))
        if score > threshold:
            self._record_failure(user, now)
            return VerifyResult(False, "no_match", score, threshold)

        if user.failures != 0:
            self.store.set_lockout(app_id, user_id, 0, 0.0)
        self.store.add_sample(app_id, user_id, "verify", emb, now)
        self.store.trim_samples(app_id, user_id, "verify", self.config.history_size)
        if self.config.adapt and score <= self.config.adapt_max_ratio * threshold:
            updated = (1 - self.config.adapt_alpha) * user.template + self.config.adapt_alpha * emb
            self.store.set_template(app_id, user_id, updated / np.linalg.norm(updated), now)
        return VerifyResult(True, "match", score, threshold)

    # ---- management -----------------------------------------------------

    def delete_user(self, app_id: str, user_id: str) -> bool:
        return self.store.delete_user(app_id, user_id)

    def status(self, app_id: str, user_id: str) -> dict | None:
        user = self.store.get_user(app_id, user_id)
        if user is None:
            return None
        return {
            "user_id": user_id,
            "status": user.status,
            "enroll_samples": self.store.count_samples(app_id, user_id, "enroll"),
            "enroll_required": self.config.enroll_samples,
            "locked": user.locked_until > self.clock(),
        }

    # ---- helpers --------------------------------------------------------

    def _embed(self, timings: np.ndarray) -> np.ndarray:
        emb = np.asarray(self.embedder(timings), dtype=np.float64)
        return emb / np.linalg.norm(emb)

    def _threshold_for(self, user) -> float:
        if user is not None and user.threshold is not None:
            return float(user.threshold)
        return self.threshold

    def _is_replay(self, app_id: str, user_id: str, emb: np.ndarray) -> bool:
        seen = self.store.get_samples(app_id, user_id)
        if seen.size == 0 or seen.shape[1] != emb.shape[0]:
            return False
        return bool(np.min(np.linalg.norm(seen - emb, axis=1)) < self.config.replay_epsilon)

    def _record_failure(self, user, now: float) -> None:
        failures = user.failures + 1
        locked_until = 0.0
        if failures >= self.config.max_failures:
            failures = 0
            locked_until = now + self.config.lockout_s
        self.store.set_lockout(user.app_id, user.user_id, failures, locked_until)
