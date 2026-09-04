"""Варианты поведения агента: случайное блуждание и нейросеть."""

import random
from abc import ABC, abstractmethod

import numpy as np

from config import (
    NEURAL_BLOCK_REVERSE,
    NEURAL_STICKY_STEPS,
    NEURAL_SWITCH_MARGIN,
    RANDOM_WALK_MAX_STEPS,
    RANDOM_WALK_MIN_STEPS,
)
from engine import MOVEMENT_ACTIONS, OPPOSITE_ACTIONS
from nn_model import FoodPolicyNetwork


class Behavior(ABC):
    """Общий интерфейс поведения: по признакам выбрать действие для движка."""

    name: str = "behavior"

    @abstractmethod
    def choose_action(self, features: np.ndarray) -> int:
        raise NotImplementedError

    def reset(self) -> None:
        """Сброс внутреннего состояния (если есть)."""
        pass


class RandomWalkBehavior(Behavior):
    """
    Выбирает направление и держит его несколько шагов подряд
    (случайно от min_steps до max_steps), затем выбирает новое.
    Только MOVEMENT_ACTIONS — без стояния.
    """

    name = "random"

    def __init__(
        self,
        min_steps: int = RANDOM_WALK_MIN_STEPS,
        max_steps: int = RANDOM_WALK_MAX_STEPS,
    ):
        self.min_steps = min_steps
        self.max_steps = max_steps
        self._action = None
        self._remaining = 0

    def reset(self) -> None:
        self._action = None
        self._remaining = 0

    def choose_action(self, features: np.ndarray) -> int:
        if self._remaining <= 0:
            self._action = random.choice(MOVEMENT_ACTIONS)
            self._remaining = random.randint(self.min_steps, self.max_steps)
        self._remaining -= 1
        return self._action


class NeuralBehavior(Behavior):
    """
    Предсказание нейросети с анти-дребезгом:
    - держит действие минимум NEURAL_STICKY_STEPS шагов;
    - меняет направление только если новый ход увереннее текущего
      на NEURAL_SWITCH_MARGIN;
    - по желанию блокирует мгновенный разворот 180°.
    Иначе argmax каждый кадр даёт «шатание» (влево-вправо и т.п.).
    """

    name = "neural"

    def __init__(
        self,
        network: FoodPolicyNetwork,
        sticky_steps: int = NEURAL_STICKY_STEPS,
        switch_margin: float = NEURAL_SWITCH_MARGIN,
        block_reverse: bool = NEURAL_BLOCK_REVERSE,
    ):
        self.network = network
        self.sticky_steps = sticky_steps
        self.switch_margin = switch_margin
        self.block_reverse = block_reverse
        self._action = None
        self._held = 0

    @property
    def is_ready(self) -> bool:
        """Сеть всегда может предсказывать; is_trained — отдельно (после fit)."""
        return True

    def reset(self) -> None:
        self._action = None
        self._held = 0

    def choose_action(self, features: np.ndarray) -> int:
        probs = self.network.predict_probs(features)
        proposed = MOVEMENT_ACTIONS[int(np.argmax(probs))]

        if self._action is None:
            self._action = proposed
            self._held = 1
            return self._action

        self._held += 1
        if self._held < self.sticky_steps:
            return self._action

        if proposed == self._action:
            return self._action

        if self.block_reverse and OPPOSITE_ACTIONS.get(self._action) == proposed:
            # разворот только если он заметно увереннее текущего
            cur_p = float(probs[MOVEMENT_ACTIONS.index(self._action)])
            new_p = float(probs[MOVEMENT_ACTIONS.index(proposed)])
            if new_p < cur_p + self.switch_margin * 1.5:
                return self._action

        cur_p = float(probs[MOVEMENT_ACTIONS.index(self._action)])
        new_p = float(probs[MOVEMENT_ACTIONS.index(proposed)])
        if new_p >= cur_p + self.switch_margin:
            self._action = proposed
            self._held = 1

        return self._action


class TrainingBehavior(RandomWalkBehavior):
    """На время fit — то же случайное блуждание, отдельное имя для дашборда."""

    name = "training"
