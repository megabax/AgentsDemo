"""Варианты поведения агента: случайное блуждание и нейросеть."""

import random
from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np

from config import (
    NEURAL_BLOCK_REVERSE,
    NEURAL_STICKY_STEPS,
    NEURAL_SWITCH_MARGIN,
    PAIN_MEMORY_SIZE,
    PAIN_RADAR_MAX_DIFF,
    PAIN_REPEAT_TO_FORBID,
    RANDOM_WALK_MAX_STEPS,
    RANDOM_WALK_MIN_STEPS,
)
from engine import MOVEMENT_ACTIONS, OPPOSITE_ACTIONS, PERPENDICULAR_ACTIONS
from nn_model import (
    FoodPolicyNetwork,
    current_radar_from_features,
    radar_pain_context,
    radar_to_features,
)


class PainContextMemory:
    """
    Память «при таком радаре это направление больно».
    Запрет ставится только после повторной боли в похожем контексте.
    """

    def __init__(
        self,
        repeat_to_forbid: int = PAIN_REPEAT_TO_FORBID,
        max_diff: float = PAIN_RADAR_MAX_DIFF,
        max_size: int = PAIN_MEMORY_SIZE,
    ):
        self.repeat_to_forbid = repeat_to_forbid
        self.max_diff = max_diff
        self.max_size = max_size
        self._entries: List[dict] = []

    def clear(self) -> None:
        self._entries.clear()

    def forbidden_count(self) -> int:
        return sum(1 for entry in self._entries if entry["forbidden"])

    def record_pain(self, radar_vec: np.ndarray, action: int) -> bool:
        """Учесть боль. True, если направление в этом контексте стало запрещённым."""
        entry = self._match(radar_vec, action)
        if entry is None:
            self._entries.append(
                {
                    "radar": radar_pain_context(radar_vec),
                    "action": action,
                    "count": 1,
                    "forbidden": self.repeat_to_forbid <= 1,
                }
            )
            self._trim()
            return self._entries[-1]["forbidden"]

        entry["count"] += 1
        entry["radar"] = radar_pain_context(radar_vec)
        if entry["count"] >= self.repeat_to_forbid:
            entry["forbidden"] = True
        return entry["forbidden"]

    def is_forbidden(self, radar_vec: np.ndarray, action: int) -> bool:
        entry = self._match(radar_vec, action)
        return entry is not None and entry["forbidden"]

    def _match(self, radar_vec: np.ndarray, action: int) -> Optional[dict]:
        best = None
        best_diff = self.max_diff
        for entry in self._entries:
            if entry["action"] != action:
                continue
            diff = _mean_abs(radar_pain_context(radar_vec), entry["radar"])
            if diff <= best_diff:
                best = entry
                best_diff = diff
        return best

    def _trim(self) -> None:
        while len(self._entries) > self.max_size:
            drop_at = next(
                (i for i, entry in enumerate(self._entries) if not entry["forbidden"]),
                0,
            )
            self._entries.pop(drop_at)


def _mean_abs(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


def _allowed_actions(pain_memory: Optional["PainContextMemory"], radar_vec, avoid: int):
    actions = []
    for action in MOVEMENT_ACTIONS:
        if action == avoid:
            continue
        if pain_memory is not None and pain_memory.is_forbidden(radar_vec, action):
            continue
        actions.append(action)
    if actions:
        return actions
    return list(PERPENDICULAR_ACTIONS.get(avoid, MOVEMENT_ACTIONS))


class Behavior(ABC):
    """Общий интерфейс поведения: по признакам выбрать действие для движка."""

    name: str = "behavior"

    @abstractmethod
    def choose_action(self, features: np.ndarray) -> int:
        raise NotImplementedError

    def reset(self) -> None:
        """Сброс внутреннего состояния (если есть)."""
        pass

    def on_pain(self, action: int, radar=None) -> None:
        """Реакция на удар о стену («боль») при данном действии и радаре."""
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
        pain_memory: Optional[PainContextMemory] = None,
    ):
        self.min_steps = min_steps
        self.max_steps = max_steps
        self.pain_memory = pain_memory
        self._action = None
        self._remaining = 0
        self._avoid = None

    def reset(self) -> None:
        self._action = None
        self._remaining = 0
        self._avoid = None

    def on_pain(self, action: int, radar=None) -> None:
        """У стены — другое направление; если контекст уже запрещён, уходим вбок."""
        self._avoid = action
        radar_vec = radar_to_features(radar) if radar is not None else None
        forbidden = (
            radar_vec is not None
            and self.pain_memory is not None
            and self.pain_memory.is_forbidden(radar_vec, action)
        )
        if forbidden:
            choices = _allowed_actions(self.pain_memory, radar_vec, action)
            side = list(PERPENDICULAR_ACTIONS.get(action, ()))
            side = [a for a in side if a in choices] or choices
            self._action = random.choice(side)
        else:
            opp = OPPOSITE_ACTIONS.get(action)
            self._action = opp if opp is not None else random.choice(MOVEMENT_ACTIONS)
        self._remaining = random.randint(self.min_steps, self.max_steps)

    def choose_action(self, features: np.ndarray) -> int:
        radar_vec = current_radar_from_features(features)
        if self._remaining <= 0 or (
            self._action is not None
            and self.pain_memory is not None
            and self.pain_memory.is_forbidden(radar_vec, self._action)
        ):
            choices = _allowed_actions(self.pain_memory, radar_vec, self._avoid or -1)
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
        pain_memory: Optional[PainContextMemory] = None,
    ):
        self.network = network
        self.sticky_steps = sticky_steps
        self.switch_margin = switch_margin
        self.block_reverse = block_reverse
        self.pain_memory = pain_memory
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

    def on_pain(self, action: int, radar=None) -> None:
        """
        Боль у стены. Первый раз — разворот.
        Повтор в похожем радаре — направление запрещено, уходим вбок.
        """
        radar_vec = radar_to_features(radar) if radar is not None else None
        forbidden = (
            radar_vec is not None
            and self.pain_memory is not None
            and self.pain_memory.is_forbidden(radar_vec, action)
        )
        if forbidden and radar_vec is not None:
            choices = _allowed_actions(self.pain_memory, radar_vec, action)
            side = list(PERPENDICULAR_ACTIONS.get(action, ()))
            side = [a for a in side if a in choices] or choices
            self._action = random.choice(side)
        else:
            opp = OPPOSITE_ACTIONS.get(action)
            if opp is None:
                self.reset()
                return
            self._action = opp
        self._held = 1
        self._escape_from_wall = True

    def _blocked(self, radar_vec: np.ndarray, action: int) -> bool:
        return (
            self.pain_memory is not None
            and self.pain_memory.is_forbidden(radar_vec, action)
        )

    def choose_action(self, features: np.ndarray) -> int:
        radar_vec = current_radar_from_features(features)
        if self._action is not None and self._blocked(radar_vec, self._action):
            self._action = self._pick_allowed(radar_vec, features, self._action)
            self._held = 1
            self._escape_from_wall = True

        # режим побега от стены — не даём сети снова упереться сразу
        if self._escape_from_wall and self._action is not None:
            self._held += 1
            if self._held < self.sticky_steps:
                return self._action
            self._escape_from_wall = False

        probs = self.network.predict_probs(features)
        proposed = MOVEMENT_ACTIONS[int(np.argmax(probs))]
        if self._blocked(radar_vec, proposed):
            proposed = self._pick_allowed(radar_vec, features, proposed)

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

    def _pick_allowed(self, radar_vec: np.ndarray, features: np.ndarray, avoid: int) -> int:
        probs = self.network.predict_probs(features)
        order = np.argsort(-probs)
        for idx in order:
            action = MOVEMENT_ACTIONS[int(idx)]
            if action == avoid or self._blocked(radar_vec, action):
                continue
            return action
        side = _allowed_actions(self.pain_memory, radar_vec, avoid)
        return random.choice(side)


class TrainingBehavior(RandomWalkBehavior):
    """На время fit — то же случайное блуждание, отдельное имя для дашборда."""

    name = "training"
