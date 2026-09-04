"""Варианты поведения агента: случайное блуждание и нейросеть."""

import random
from abc import ABC, abstractmethod

import numpy as np

from config import RANDOM_WALK_MAX_STEPS, RANDOM_WALK_MIN_STEPS
from engine import MOVEMENT_ACTIONS
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
    """Действие по предсказанию нейросети (в т.ч. до первого fit — случайные веса)."""

    name = "neural"

    def __init__(self, network: FoodPolicyNetwork):
        self.network = network

    @property
    def is_ready(self) -> bool:
        """Сеть всегда может предсказывать; is_trained — отдельно (после fit)."""
        return True

    def choose_action(self, features: np.ndarray) -> int:
        return self.network.predict_action(features)


class TrainingBehavior(RandomWalkBehavior):
    """На время fit — то же случайное блуждание, отдельное имя для дашборда."""

    name = "training"
