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

    def on_pain(self, action: int) -> None:
        """Реакция на удар о стену («боль») при данном действии."""
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
        self._avoid = None

    def reset(self) -> None:
        self._action = None
        self._remaining = 0
        self._avoid = None

    def on_pain(self, action: int) -> None:
        """У стены — сразу выбрать другое направление (предпочесть разворот)."""
        self._avoid = action
        opp = OPPOSITE_ACTIONS.get(action)
        if opp is not None:
            self._action = opp
        else:
            choices = [a for a in MOVEMENT_ACTIONS if a != action]
            self._action = random.choice(choices) if choices else random.choice(MOVEMENT_ACTIONS)
        self._remaining = random.randint(self.min_steps, self.max_steps)

    def choose_action(self, features: np.ndarray) -> int:
        if self._remaining <= 0:
            choices = list(MOVEMENT_ACTIONS)
            if self._avoid in choices and len(choices) > 1:
                choices = [a for a in choices if a != self._avoid]
            self._action = random.choice(choices)
            self._remaining = random.randint(self.min_steps, self.max_steps)
            self._avoid = None
        self._remaining -= 1
        return self._action


class NeuralBehavior(Behavior):
    """
    Предсказание нейросети с анти-дребезгом и реакцией на «боль» у стены.
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
        self._escape_from_wall = False

    @property
    def is_ready(self) -> bool:
        return True

    def reset(self) -> None:
        self._action = None
        self._held = 0
        self._escape_from_wall = False

    def on_pain(self, action: int) -> None:
        """
        Боль у стены: сбросить sticky и уйти в противоположную сторону
        на ближайшие sticky_steps (разворот от стены).
        """
        opp = OPPOSITE_ACTIONS.get(action)
        if opp is None:
            self.reset()
            return
        self._action = opp
        self._held = 1
        self._escape_from_wall = True

    def choose_action(self, features: np.ndarray) -> int:
        # режим побега от стены — не даём сети снова упереться сразу
        if self._escape_from_wall and self._action is not None:
            self._held += 1
            if self._held < self.sticky_steps:
                return self._action
            self._escape_from_wall = False

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
