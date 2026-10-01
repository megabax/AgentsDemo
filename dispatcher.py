"""Диспетчер: переключает объекты Behavior (neural ↔ random)."""

from collections import deque
from typing import Optional

import numpy as np

from behaviors import (
    Behavior,
    NeuralBehavior,
    PainContextMemory,
    RandomWalkBehavior,
    TrainingBehavior,
)
from config import (
    CYCLE_MAX_PERIOD,
    CYCLE_MIN_REPEATS,
    DISPATCH_STALE_ATTEMPTS,
    NEURAL_STICKY_STEPS,
    NEURAL_SWITCH_MARGIN,
    PAIN_REPEAT_TO_FORBID,
    RANDOM_WALK_MAX_STEPS,
    RANDOM_WALK_MIN_STEPS,
)
from experience import Attempt, AttemptOutcome
from nn_model import FoodPolicyNetwork, radar_to_features


class MovementCycleDetector:
    """Ищет повтор одного и того же куска траектории."""

    def __init__(
        self,
        min_repeats: int = CYCLE_MIN_REPEATS,
        max_period: int = CYCLE_MAX_PERIOD,
    ):
        self.min_repeats = min_repeats
        self.max_period = max_period
        self._positions: deque = deque(maxlen=min_repeats * max_period)

    def clear(self) -> None:
        self._positions.clear()

    def push(self, x: float, y: float) -> bool:
        self._positions.append((int(round(x)), int(round(y))))
        return self._is_cycling()

    def _is_cycling(self) -> bool:
        seq = list(self._positions)
        n = len(seq)
        for period in range(2, self.max_period + 1):
            need = period * self.min_repeats
            if n < need:
                continue
            window = seq[-need:]
            pattern = window[:period]
            if len(set(pattern)) < 2:
                continue
            if all(window[i] == pattern[i % period] for i in range(need)):
                return True
        return False


class ModeDispatcher:
    """
    Старт с нейросети. На random переключается только если
    текущее поведение долго не находит еду (stale attempts).
    """

    def __init__(
        self,
        network: FoodPolicyNetwork,
        stale_attempts: int = DISPATCH_STALE_ATTEMPTS,
        behavior_block=None,
    ):
        self.network = network
        self.stale_limit = stale_attempts
        if behavior_block is None:
            pain_repeat = PAIN_REPEAT_TO_FORBID
            cycle_repeats = CYCLE_MIN_REPEATS
            walk_min = RANDOM_WALK_MIN_STEPS
            walk_max = RANDOM_WALK_MAX_STEPS
            sticky = NEURAL_STICKY_STEPS
            margin = NEURAL_SWITCH_MARGIN
        else:
            pain_repeat = behavior_block.gene("pain_repeat_to_forbid").value
            cycle_repeats = behavior_block.gene("cycle_min_repeats").value
            walk_min = behavior_block.gene("random_walk_min_steps").value
            walk_max = behavior_block.gene("random_walk_max_steps").value
            sticky = behavior_block.gene("neural_sticky_steps").value
            margin = behavior_block.gene("neural_switch_margin").value
        self.pain_memory = PainContextMemory(repeat_to_forbid=pain_repeat)
        self.cycle_detector = MovementCycleDetector(min_repeats=cycle_repeats)
        self.cycle_switches = 0
        self.random_behavior = RandomWalkBehavior(
            min_steps=walk_min,
            max_steps=walk_max,
            pain_memory=self.pain_memory,
        )
        self.neural_behavior = NeuralBehavior(
            network,
            sticky_steps=sticky,
            switch_margin=margin,
            pain_memory=self.pain_memory,
        )
        self.training_behavior = TrainingBehavior(
            min_steps=walk_min,
            max_steps=walk_max,
            pain_memory=self.pain_memory,
        )
        self.current: Behavior = self.neural_behavior
        self.stale_attempts = 0
        self.switch_count = 0

    @property
    def mode(self):
        """Совместимость: имя текущего поведения (Enum-like .value)."""
        return _ModeName(self.current.name)

    def reset(self) -> None:
        self.pain_memory.clear()
        self.cycle_detector.clear()
        self.cycle_switches = 0
        self.random_behavior.reset()
        self.training_behavior.reset()
        self.current = self.neural_behavior
        self.stale_attempts = 0
        self.switch_count = 0

    def choose_action(self, features: np.ndarray) -> int:
        return self.current.choose_action(features)

    def on_pain(self, action: int, radar=None) -> None:
        """Запомнить боль в контексте радара и передать текущему поведению."""
        if radar is not None:
            self.pain_memory.record_pain(radar_to_features(radar), action)
        self.current.on_pain(action, radar)

    def note_position(self, x: float, y: float) -> Optional[str]:
        """
        В режиме neural: если один и тот же кусок пути повторился
        более трёх раз — перейти на случайное блуждание.
        Обратно на neural — прежние правила (еда или застой random).
        """
        if self.current is not self.neural_behavior:
            return None
        if not self.cycle_detector.push(x, y):
            return None
        self.set_random()
        self.cycle_detector.clear()
        self.cycle_switches += 1
        self.switch_count += 1
        return "cycle->random"

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
        self.cycle_detector.clear()
        self.current = self.neural_behavior
        self.stale_attempts = 0

    def set_random(self) -> None:
        self.cycle_detector.clear()
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
            "pain_forbids": self.pain_memory.forbidden_count(),
            "cycle_switches": self.cycle_switches,
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
