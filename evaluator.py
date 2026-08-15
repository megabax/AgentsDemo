"""Оценщик (legacy): логика переключения перенесена в ModeDispatcher."""

from collections import deque
from typing import Deque, Optional

from experience import Attempt, AttemptOutcome


class PerformanceEvaluator:
    """Устарело: см. dispatcher.ModeDispatcher (stale attempts)."""

    def __init__(self, window: int = 12, min_samples: int = 5, min_success_rate: float = 0.25):
        self.window = window
        self.min_samples = min_samples
        self.min_success_rate = min_success_rate
        self._results: Deque[bool] = deque(maxlen=window)

    def reset(self) -> None:
        self._results.clear()

    def record_attempt(self, attempt: Attempt) -> None:
        if attempt.outcome == AttemptOutcome.ABORTED:
            return
        self._results.append(attempt.outcome == AttemptOutcome.SUCCESS)

    @property
    def sample_count(self) -> int:
        return len(self._results)

    @property
    def success_rate(self) -> Optional[float]:
        if not self._results:
            return None
        return sum(self._results) / len(self._results)

    def should_switch(self) -> bool:
        if self.sample_count < self.min_samples:
            return False
        rate = self.success_rate
        return rate is not None and rate < self.min_success_rate
