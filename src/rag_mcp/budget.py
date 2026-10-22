"""In-memory call budget: a per-minute sliding window plus an optional daily cap.

State is per server process: it resets on restart and is not shared between instances
(documented limitation).
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable
from typing import Literal

Refusal = Literal["minute", "day"]


class Budget:
    def __init__(self, per_minute: int, per_day: int | None = None, clock: Callable[[], float] = time.time):
        if per_minute < 1 or (per_day is not None and per_day < 1):
            raise ValueError("limits must be positive")
        self.per_minute = per_minute
        self.per_day = per_day
        self._clock = clock
        self._window: deque[float] = deque()
        self._day = self._today()
        self._day_count = 0

    def _today(self) -> int:
        return int(self._clock() // 86400)

    def try_acquire(self) -> Refusal | None:
        """Consumes one call if allowed. Returns None when allowed, else the reason."""
        now = self._clock()
        today = self._today()
        if today != self._day:
            self._day, self._day_count = today, 0
        if self.per_day is not None and self._day_count >= self.per_day:
            return "day"
        while self._window and now - self._window[0] >= 60:
            self._window.popleft()
        if len(self._window) >= self.per_minute:
            return "minute"
        self._window.append(now)
        self._day_count += 1
        return None
