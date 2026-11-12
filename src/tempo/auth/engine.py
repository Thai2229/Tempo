"""Enrollment and verification.

The engine works on timing arrays only. It never sees the password: the
integrating app checks the password itself and asks tempo whether the
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

Two optional extensions (both off by default, so the plain Embedder engine is
unchanged):
  - per_user: PerUserThresholdParams (tempo.thresholds). When enrollment
    completes, the user's own threshold is computed from their enrollment
    attempts and stored. If it cannot be computed (too few attempts, bad
    numbers) the user keeps the global threshold. The "inconsistent
    enrollment" check always uses the GLOBAL threshold: it runs before the
    user has a threshold of their own, and the conservative choice is that a
    loose personal threshold must never excuse a sloppy enrollment.
  - scorer: a UserScorer (scaled Manhattan / Mahalanobis, tempo.auth.scorers)
    instead of an Embedder. The "representation" of an attempt is then its
    raw timing vector in seconds; the store keeps these as samples, the fitted
    ScorerState is stored as JSON, and the score is scorer.score(state, vector).
    Template adaptation is off in this mode: a template can be nudged toward a
    new attempt, but mean/scale/precision statistics cannot be updated that
    way without refitting on a mix of enrollment and login vectors, which
    would let a user's own (possibly attacker-influenced) logins widen the
    model unreviewed.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np

from tempo.auth.scorers import ScorerState, UserScorer, enrollment_loo_scores, timing_vector
from tempo.auth.store import TemplateStore
from tempo.auth.validation import validate_timings
from tempo.evaluate import make_template
from tempo.thresholds import PerUserThresholdParams, embedding_threshold, user_threshold


class Embedder(Protocol):
    def __call__(self, timings: np.ndarray) -> np.ndarray:
        """(n, 3) validated timings -> unit-length embedding."""


@dataclass
class EngineConfig:
    enroll_samples: int = 10
    replay_epsilon: float = 1e-4  # embedding distance below which an attempt is a replay
    # Scorer mode: L2 distance between timing vectors (seconds) below which an attempt is a replay.
    # 1e-4 s is 0.1 ms over the whole vector (D = 3n - 2 values): far below the 1 ms resolution of
    # browser key timestamps, so no two real attempts get that close, while a copied request
    # (identical up to float32 storage, ~1e-8) is caught. Raise it if the client adds jitter.
    replay_epsilon_s: float = 1e-4
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
    # "match" | "no_match" | "not_enrolled" | "locked" | "length_mismatch" | "replay"
    # | "scorer_error" (scorer mode: stored state unreadable or unusable; fails closed, not counted as a failure)
    reason: str
    score: float | None  # distance to template, lower = more similar
    threshold: float


class TempoEngine:
    def __init__(
        self,
        embedder: Embedder | None,
        store: TemplateStore,
        threshold: float,
        config: EngineConfig | None = None,
        clock: Callable[[], float] = time.time,
        per_user: PerUserThresholdParams | None = None,
        scorer: UserScorer | None = None,
    ):
        if config is None:
            config = EngineConfig()
        if (embedder is None) == (scorer is None):
            raise ValueError("pass exactly one of embedder or scorer")
        if scorer is not None and config.enroll_samples < 3:
            raise ValueError("scorer mode needs enroll_samples >= 3 (leave-one-out consistency check)")
        self.embedder = embedder
        self.store = store
        self.threshold = float(threshold)
        self.config = config
        self.clock = clock
        self.per_user = per_user
        self.scorer = scorer

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

        emb = self._represent(t)
        if self._is_replay(app_id, user_id, emb):
            count = self.store.count_samples(app_id, user_id, "enroll")
            return EnrollResult("rejected", count, required, "replay")

        self.store.add_sample(app_id, user_id, "enroll", emb, now)
        count = self.store.count_samples(app_id, user_id, "enroll")
        if count < required:
            return EnrollResult("in_progress", count, required)

        samples = self.store.get_samples(app_id, user_id, "enroll")
        if self.scorer is not None:
            return self._complete_scorer_enrollment(app_id, user_id, user, samples, now)

        template = make_template(samples)
        spread = float(np.median(np.linalg.norm(samples - template, axis=1)))
        # Global threshold on purpose (see module docstring): `user` has no personal threshold yet.
        if spread > self._threshold_for(user):
            self.store.clear_samples(app_id, user_id, "enroll")
            return EnrollResult("rejected", 0, required, "inconsistent")

        self._store_user_threshold(app_id, user_id, lambda: embedding_threshold(samples, self.per_user))
        self.store.set_template(app_id, user_id, template, now)
        return EnrollResult("complete", count, required)

    def _complete_scorer_enrollment(self, app_id: str, user_id: str, user, samples: np.ndarray, now: float) -> EnrollResult:
        required = self.config.enroll_samples
        loo = enrollment_loo_scores(self.scorer, samples)
        spread = float(np.median(loo))
        # Same rule as the embedder path: the global threshold, never the user's own.
        if not np.isfinite(spread) or spread > self._threshold_for(user):
            self.store.clear_samples(app_id, user_id, "enroll")
            return EnrollResult("rejected", 0, required, "inconsistent")

        state = self.scorer.fit(samples)
        self._store_user_threshold(app_id, user_id, lambda: user_threshold(loo, self.per_user))
        self.store.set_scorer_state(app_id, user_id, state.to_json(), now)
        return EnrollResult("complete", len(samples), required)

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

        emb = self._represent(t)
        if self._is_replay(app_id, user_id, emb):
            self._record_failure(user, now)
            return VerifyResult(False, "replay", None, threshold)

        if self.scorer is None:
            score = float(np.linalg.norm(emb - user.template))
        else:
            try:
                score = self.scorer.score(ScorerState.from_json(user.scorer_state), emb)
            except ValueError:
                # Corrupt, missing or mismatched state is a server-side fault: fail closed with a
                # reason instead of a 500, and do not count it against the user's lockout.
                return VerifyResult(False, "scorer_error", None, threshold)
        # a non-finite score must never be accepted (NaN > threshold is False)
        if not np.isfinite(score) or score > threshold:
            self._record_failure(user, now)
            return VerifyResult(False, "no_match", score, threshold)

        if user.failures != 0:
            self.store.set_lockout(app_id, user_id, 0, 0.0)
        self.store.add_sample(app_id, user_id, "verify", emb, now)
        self.store.trim_samples(app_id, user_id, "verify", self.config.history_size)
        if self.scorer is None and self.config.adapt and score <= self.config.adapt_max_ratio * threshold:
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

    def _represent(self, timings: np.ndarray) -> np.ndarray:
        """What the engine stores and compares: an embedding, or in scorer mode the timing vector."""
        if self.scorer is None:
            return self._embed(timings)
        return timing_vector(timings)

    def _store_user_threshold(self, app_id: str, user_id: str, compute: Callable[[], float]) -> None:
        """Persist a per-user threshold if params are set and it is finite and positive.

        On any failure (e.g. fewer than 3 enrollment attempts) the user keeps the global threshold.
        """
        if self.per_user is None:
            return
        try:
            value = float(compute())
        except ValueError:
            return
        if np.isfinite(value) and value > 0:
            self.store.set_threshold(app_id, user_id, value)

    def _threshold_for(self, user) -> float:
        if user is not None and user.threshold is not None:
            return float(user.threshold)
        return self.threshold

    def _is_replay(self, app_id: str, user_id: str, emb: np.ndarray) -> bool:
        seen = self.store.get_samples(app_id, user_id)
        if seen.size == 0 or seen.shape[1] != emb.shape[0]:
            return False
        epsilon = self.config.replay_epsilon
        if self.scorer is not None:
            epsilon = self.config.replay_epsilon_s
        return bool(np.min(np.linalg.norm(seen - emb, axis=1)) < epsilon)

    def _record_failure(self, user, now: float) -> None:
        failures = user.failures + 1
        locked_until = 0.0
        if failures >= self.config.max_failures:
            failures = 0
            locked_until = now + self.config.lockout_s
        self.store.set_lockout(user.app_id, user.user_id, failures, locked_until)
