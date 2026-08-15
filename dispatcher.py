"""Диспетчер: кто управляет движком — random или нейросеть."""

import random
from enum import Enum
from typing import Optional

import numpy as np

from config import DISPATCH_STALE_ATTEMPTS
from engine import ALL_ACTIONS
from experience import Attempt, AttemptOutcome
from nn_model import FoodPolicyNetwork


class ControllerMode(str, Enum):
    NEURAL = "neural"
    RANDOM = "random"
    TRAINING = "training"


class ModeDispatcher:
    """
    Переключает random ↔ neural, если текущий метод долго не даёт еды.
    Нейросеть сама движок не трогает — только предлагает действие диспетчеру.
    """

    def __init__(
        self,
        network: FoodPolicyNetwork,
        stale_attempts: int = DISPATCH_STALE_ATTEMPTS,
    ):
        self.network = network
        self.stale_limit = stale_attempts
        self.mode = ControllerMode.RANDOM
        self.stale_attempts = 0  # подряд попыток без еды в текущем режиме
        self.switch_count = 0

    def reset(self) -> None:
        self.mode = ControllerMode.RANDOM
        self.stale_attempts = 0
        self.switch_count = 0

    def choose_action(self, features: np.ndarray) -> int:
        if self.mode == ControllerMode.TRAINING:
            return random.choice(ALL_ACTIONS)

        if self.mode == ControllerMode.NEURAL and self.network.is_trained:
            return self.network.predict_action(features)

        # нет обученной сети или режим random
        if self.mode == ControllerMode.NEURAL and not self.network.is_trained:
            self.mode = ControllerMode.RANDOM
        return random.choice(ALL_ACTIONS)

    def on_attempt_end(self, attempt: Attempt) -> Optional[str]:
        """
        Учёт исхода. Возвращает причину переключения режима или None.
        """
        if attempt.outcome == AttemptOutcome.ABORTED:
            return None

        if attempt.outcome == AttemptOutcome.SUCCESS:
            self.stale_attempts = 0
            return None

        # failure — метод «долго не даёт результат»
        self.stale_attempts += 1
        if self.stale_attempts >= self.stale_limit:
            return self._switch_method()
        return None

    def _switch_method(self) -> str:
        old = self.mode
        if self.mode == ControllerMode.NEURAL:
            self.mode = ControllerMode.RANDOM
        elif self.mode == ControllerMode.RANDOM:
            if self.network.is_trained:
                self.mode = ControllerMode.NEURAL
            else:
                # сети ещё нет — остаёмся в random, сбрасываем счётчик
                self.stale_attempts = 0
                return "stay_random_untrained"
        else:
            self.mode = ControllerMode.RANDOM

        self.stale_attempts = 0
        self.switch_count += 1
        return f"{old.value}->{self.mode.value}"

    def set_training(self) -> None:
        self.mode = ControllerMode.TRAINING

    def set_neural(self) -> None:
        self.mode = ControllerMode.NEURAL
        self.stale_attempts = 0

    def set_random(self) -> None:
        self.mode = ControllerMode.RANDOM
        self.stale_attempts = 0

    def status_dict(self) -> dict:
        return {
            "mode": self.mode.value,
            "stale_attempts": self.stale_attempts,
            "stale_limit": self.stale_limit,
            "switch_count": self.switch_count,
            "network_trained": self.network.is_trained,
        }
