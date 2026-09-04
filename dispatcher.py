"""Диспетчер: переключает объекты Behavior (neural ↔ random)."""

from typing import Optional

import numpy as np

from behaviors import (
    Behavior,
    NeuralBehavior,
    RandomWalkBehavior,
    TrainingBehavior,
)
from config import DISPATCH_STALE_ATTEMPTS
from experience import Attempt, AttemptOutcome
from nn_model import FoodPolicyNetwork


class ModeDispatcher:
    """
    Старт с нейросети. На random переключается только если
    текущее поведение долго не находит еду (stale attempts).
    """

    def __init__(
        self,
        network: FoodPolicyNetwork,
        stale_attempts: int = DISPATCH_STALE_ATTEMPTS,
    ):
        self.network = network
        self.stale_limit = stale_attempts
        self.random_behavior = RandomWalkBehavior()
        self.neural_behavior = NeuralBehavior(network)
        self.training_behavior = TrainingBehavior()
        self.current: Behavior = self.neural_behavior
        self.stale_attempts = 0
        self.switch_count = 0

    @property
    def mode(self):
        """Совместимость: имя текущего поведения (Enum-like .value)."""
        return _ModeName(self.current.name)

    def reset(self) -> None:
        self.random_behavior.reset()
        self.training_behavior.reset()
        self.current = self.neural_behavior
        self.stale_attempts = 0
        self.switch_count = 0

    def choose_action(self, features: np.ndarray) -> int:
        return self.current.choose_action(features)

    def on_pain(self, action: int) -> None:
        """Передать «боль» от стены текущему поведению."""
        self.current.on_pain(action)

    def on_attempt_end(self, attempt: Attempt) -> Optional[str]:
        if attempt.outcome == AttemptOutcome.ABORTED:
            return None

        if attempt.outcome == AttemptOutcome.SUCCESS:
            self.stale_attempts = 0
            # после успеха на random можно вернуться к нейросети
            if self.current is self.random_behavior:
                return self._switch_to_neural(reason="food_then_neural")
            return None

        self.stale_attempts += 1
        if self.stale_attempts >= self.stale_limit:
            return self._switch_method()
        return None

    def _switch_to_neural(self, reason: str = "to_neural") -> str:
        old = self.current.name
        self.current = self.neural_behavior
        self.stale_attempts = 0
        self.switch_count += 1
        self.current.reset()
        return f"{old}->{self.current.name}:{reason}"

    def _switch_method(self) -> str:
        """При застое: neural → random; random → neural."""
        old = self.current.name
        if self.current is self.neural_behavior or self.current is self.training_behavior:
            self.random_behavior.reset()
            self.current = self.random_behavior
        elif self.current is self.random_behavior:
            self.current = self.neural_behavior
        else:
            self.current = self.neural_behavior

        self.stale_attempts = 0
        self.switch_count += 1
        self.current.reset()
        return f"{old}->{self.current.name}"

    def set_training(self) -> None:
        self.training_behavior.reset()
        self.current = self.training_behavior

    def set_neural(self) -> None:
        self.current = self.neural_behavior
        self.stale_attempts = 0

    def set_random(self) -> None:
        self.random_behavior.reset()
        self.current = self.random_behavior
        self.stale_attempts = 0

    def status_dict(self) -> dict:
        return {
            "mode": self.current.name,
            "stale_attempts": self.stale_attempts,
            "stale_limit": self.stale_limit,
            "switch_count": self.switch_count,
            "network_trained": self.network.is_trained,
            "behavior": self.current.__class__.__name__,
        }


class _ModeName:
    """Обёртка с .value, чтобы agent.controller_mode.value продолжал работать."""

    def __init__(self, name: str):
        self.value = name
        self.name = name.upper()

    def __eq__(self, other) -> bool:
        if isinstance(other, _ModeName):
            return self.value == other.value
        if isinstance(other, str):
            return self.value == other
        return NotImplemented

    def __repr__(self) -> str:
        return f"Mode({self.value!r})"
