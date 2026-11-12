"""In-memory sliding-window rate limiter (per process; not shared across workers)."""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from typing import Callable, Hashable


class SlidingWindowLimiter:
    def __init__(self, limit: int, window_s: float, clock: Callable[[], float] = time.monotonic):
        self.limit = limit
        self.window_s = window_s
        self.clock = clock
        self._hits: dict[Hashable, deque[float]] = {}
        self._lock = threading.Lock()

    def check(self, keys: list[Hashable]) -> int | None:
        """Record one request against every key.

        Returns None if allowed, else the seconds to wait (>= 1). A refused request is
        not recorded, so a blocked client does not extend its own block.
        """
        now = self.clock()
        with self._lock:
            if len(self._hits) > 10000:
                self._sweep(now)
            wait = 0.0
            for key in keys:
                hits = self._hits.get(key)
                if hits is None:
                    continue
                while hits and hits[0] <= now - self.window_s:
                    hits.popleft()
                if len(hits) >= self.limit:
                    wait = max(wait, hits[0] + self.window_s - now)
            if wait > 0:
                return max(1, math.ceil(wait))
            for key in keys:
                self._hits.setdefault(key, deque()).append(now)
        return None

    def _sweep(self, now: float) -> None:
        stale = [k for k, h in self._hits.items() if not h or h[-1] <= now - self.window_s]
        for k in stale:
            del self._hits[k]
