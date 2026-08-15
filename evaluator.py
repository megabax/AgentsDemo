"""Оценщик качества нейросети: решать, уходить ли в случайное блуждание."""

from collections import deque
from typing import Deque, Optional

from config import EVAL_MIN_SAMPLES, EVAL_MIN_SUCCESS_RATE, EVAL_WINDOW
from experience import Attempt, AttemptOutcome


class PerformanceEvaluator:
    """
    Смотрит на последние нейросетевые попытки.
    Если доля успехов ниже порога — рекомендовать random walk.
    """

    def __init__(
        self,
        window: int = EVAL_WINDOW,
        min_samples: int = EVAL_MIN_SAMPLES,
        min_success_rate: float = EVAL_MIN_SUCCESS_RATE,
    ):
        self.window = window
        self.min_samples = min_samples
        self.min_success_rate = min_success_rate
        self._neural_results: Deque[bool] = deque(maxlen=window)

    def reset(self) -> None:
        self._neural_results.clear()

    def record_attempt(self, attempt: Attempt) -> None:
        if attempt.source_mode != "neural":
            return
        if attempt.outcome == AttemptOutcome.ABORTED:
            return
        self._neural_results.append(attempt.outcome == AttemptOutcome.SUCCESS)

    @property
    def sample_count(self) -> int:
        return len(self._neural_results)

    @property
    def success_rate(self) -> Optional[float]:
        if not self._neural_results:
            return None
        return sum(self._neural_results) / len(self._neural_results)

    def should_switch_to_random(self) -> bool:
        if self.sample_count < self.min_samples:
            return False
        rate = self.success_rate
        if rate is None:
            return False
        return rate < self.min_success_rate

    def status_dict(self) -> dict:
        return {
            "eval_samples": self.sample_count,
            "eval_success_rate": self.success_rate,
            "eval_threshold": self.min_success_rate,
            "switch_to_random": self.should_switch_to_random(),
        }
