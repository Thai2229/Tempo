"""Synthetic typists, for tests and demos without real biometric data.

Each typist has a characteristic mean hold time per key and mean press-to-press
latency per key pair; attempts add log-normal noise around those means. That
reproduces the two properties the model relies on: attempts by one person are
similar, and different people differ in a key-specific (not just overall-speed)
way.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from keyauth.config import MS_PER_SECOND, SUBMIT_KEY
from keyauth.events import KeyEvent, RawEvent


@dataclass(frozen=True)
class TypistProfile:
    keys: tuple[str, ...]
    hold_mean: np.ndarray  # (n,) seconds
    dd_mean: np.ndarray  # (n-1,) seconds
    sigma: float  # log-space noise of a single attempt


def make_typist(phrase: str, rng: np.random.Generator, add_submit: bool = True) -> TypistProfile:
    keys = list(phrase)
    if add_submit:
        keys.append(SUBMIT_KEY)
    n = len(keys)

    speed = rng.lognormal(mean=0.0, sigma=0.3)  # >1 = slower typist overall
    hold_mean = 0.09 * np.sqrt(speed) * np.exp(rng.normal(0.0, 0.25, size=n))
    dd_mean = 0.20 * speed * np.exp(rng.normal(0.0, 0.40, size=n - 1))
    sigma = float(rng.uniform(0.10, 0.25))
    return TypistProfile(keys=tuple(keys), hold_mean=hold_mean, dd_mean=dd_mean, sigma=sigma)


def simulate_attempt(profile: TypistProfile, rng: np.random.Generator, t0_ms: float = 0.0) -> list[KeyEvent]:
    """One typing attempt as paired keystrokes (milliseconds)."""
    n = len(profile.keys)
    hold = profile.hold_mean * np.exp(rng.normal(0.0, profile.sigma, size=n))
    dd = profile.dd_mean * np.exp(rng.normal(0.0, profile.sigma, size=n - 1))

    down = np.concatenate([[0.0], np.cumsum(dd)]) * MS_PER_SECOND + t0_ms
    up = down + hold * MS_PER_SECOND
    return [KeyEvent(key=k, down=float(d), up=float(u)) for k, d, u in zip(profile.keys, down, up)]


def to_raw_events(keystrokes: list[KeyEvent]) -> list[RawEvent]:
    """Expand keystrokes into a browser-like raw down/up stream.

    Upper-case letters get a surrounding Shift press, so the stream exercises
    modifier filtering the same way real captures do.
    """
    raw: list[RawEvent] = []
    for ks in keystrokes:
        code = _code_for(ks.key)
        if len(ks.key) == 1 and ks.key.isupper():
            raw.append(RawEvent(key="Shift", type="down", t=ks.down - 30.0, code="ShiftLeft"))
            raw.append(RawEvent(key="Shift", type="up", t=ks.up + 5.0, code="ShiftLeft"))
        raw.append(RawEvent(key=ks.key, type="down", t=ks.down, code=code))
        raw.append(RawEvent(key=ks.key, type="up", t=ks.up, code=code))
    raw.sort(key=lambda e: e.t)
    return raw


def _code_for(key: str) -> str:
    if len(key) == 1 and key.isalpha():
        return "Key" + key.upper()
    if len(key) == 1 and key.isdigit():
        return "Digit" + key
    if key == ".":
        return "Period"
    return key
