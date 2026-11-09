"""Validation of timing arrays coming from untrusted clients.

The API receives only timings (never characters), so this is the one place
that decides whether an attempt is well-formed:
  - shape (n, 3) with 2 <= n <= MAX_KEYS, all finite
  - hold in [0, MAX_HOLD_S], dd in [0, MAX_LATENCY_S]
  - dd == hold + ud for every keystroke but the last (they come from the same
    two timestamps, so a mismatch means a buggy or fabricated client)
The last row's dd/ud are undefined and are zeroed.
"""

from __future__ import annotations

import numpy as np

from keyauth.config import DD_TOLERANCE_S, MAX_HOLD_S, MAX_KEYS, MAX_LATENCY_S, N_CHANNELS


class InvalidTimings(ValueError):
    pass


def validate_timings(timings) -> np.ndarray:
    try:
        arr = np.array(timings, dtype=np.float64)
    except (TypeError, ValueError) as e:
        raise InvalidTimings(f"timings must be a list of [hold, dd, ud] rows: {e}") from None

    if arr.ndim != 2 or arr.shape[1] != N_CHANNELS or arr.shape[0] < 2 or arr.shape[0] > MAX_KEYS:
        raise InvalidTimings(f"timings must have shape (n, 3) with 2 <= n <= {MAX_KEYS}, got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise InvalidTimings("timings must be finite numbers")

    hold = arr[:, 0]
    dd = arr[:-1, 1]
    ud = arr[:-1, 2]
    if np.any(hold < 0) or np.any(hold > MAX_HOLD_S) or np.any(dd < 0) or np.any(dd > MAX_LATENCY_S):
        raise InvalidTimings("timing values out of range (are they in seconds?)")
    if np.any(np.abs(dd - (hold[:-1] + ud)) > DD_TOLERANCE_S):
        raise InvalidTimings("inconsistent timings: dd must equal hold + ud")

    arr[-1, 1:] = 0.0
    return arr.astype(np.float32)
