"""Агенты ИИ: случайные, заглушки и нейросетевой контроллер с переобучением."""

import random
from enum import Enum
from typing import Optional

from config import (
    ATTEMPT_MAX_STEPS,
    EVAL_MIN_SAMPLES,
    EXPERIENCE_KEEP_FRACTION,
    TRAIN_EVERY_N_NEURAL_FOODS,
    TRAIN_MIN_SUCCESS_STEPS,
)
from engine import ALL_ACTIONS, ACTION_STAY
from evaluator import PerformanceEvaluator
from experience import (
    Attempt,
    AttemptHistory,
    AttemptOutcome,
    ExperienceHistory,
    ExperienceStep,
    RadarReading,
)
from nn_model import FoodPolicyNetwork


class ControllerMode(str, Enum):
    NEURAL = "neural"
    RANDOM = "random"
    TRAINING = "training"


class BaseAgent:
    """
    Каркас агента: идёт к еде по радару через попытки (несколько шагов → исход).
    Покадровый опыт + журнал закрытых Attempt.
    """

    def __init__(
        self,
        history: Optional[ExperienceHistory] = None,
        attempt_history: Optional[AttemptHistory] = None,
        max_steps_per_attempt: int = ATTEMPT_MAX_STEPS,
    ):
        self.history = history if history is not None else ExperienceHistory()
        self.attempt_history = (
            attempt_history if attempt_history is not None else AttemptHistory()
        )
        self.max_steps_per_attempt = max_steps_per_attempt
        self.step_index = 0
        self.attempt_index = 0
        self.current_attempt: Optional[Attempt] = None
        self.controller_mode = ControllerMode.RANDOM

    def reset(self) -> None:
        self.history.clear()
        self.attempt_history.clear()
        self.step_index = 0
        self.attempt_index = 0
        self.current_attempt = None

    def choose_direction(self, radar: RadarReading) -> Optional[int]:
        return None

    def begin_attempt(self, radar: RadarReading) -> Attempt:
        if self.current_attempt is not None and self.current_attempt.is_open:
            self.finish_attempt(AttemptOutcome.ABORTED)

        attempt = Attempt(
            attempt_index=self.attempt_index,
            max_steps=self.max_steps_per_attempt,
            intended_direction=self.choose_direction(radar),
            initial_radar=radar,
            source_mode=self.controller_mode.value,
        )
        self.attempt_index += 1
        self.current_attempt = attempt
        return attempt

    def act(self, radar: RadarReading, state: Optional[dict] = None) -> int:
        raise NotImplementedError

    def observe(
        self,
        radar: RadarReading,
        action: int,
        food_gained: bool,
        food_count: int = 0,
    ) -> ExperienceStep:
        if self.current_attempt is None or not self.current_attempt.is_open:
            self.begin_attempt(radar)

        step = self.history.record(
            step_index=self.step_index,
            radar_reading=radar,
            action=action,
            food_gained=food_gained,
            food_count=food_count,
        )
        self.current_attempt.add_step(step)
        self.step_index += 1

        if food_gained:
            self.finish_attempt(AttemptOutcome.SUCCESS)
        elif self.current_attempt.length >= self.current_attempt.max_steps:
            self.finish_attempt(AttemptOutcome.FAILURE)

        return step

    def finish_attempt(self, outcome: AttemptOutcome) -> Optional[Attempt]:
        if self.current_attempt is None or not self.current_attempt.is_open:
            return None

        attempt = self.current_attempt
        attempt.close(outcome)
        self.attempt_history.add(attempt)
        self.on_attempt_end(attempt)
        self.current_attempt = None
        return attempt

    def on_attempt_end(self, attempt: Attempt) -> None:
        self.learn_from_attempt(attempt)

    def learn_from_attempt(self, attempt: Attempt) -> None:
        pass

    def learn(self) -> None:
        pass

    def on_episode_end(self) -> None:
        if self.current_attempt is not None and self.current_attempt.is_open:
            self.finish_attempt(AttemptOutcome.ABORTED)
        self.learn()

    def dashboard_stats(self) -> dict:
        return {
            "mode": self.controller_mode.value,
            "experience_steps": len(self.history),
            "attempts": len(self.attempt_history),
            "successes": len(self.attempt_history.successes()),
            "failures": len(self.attempt_history.failures()),
        }


class DummyAgent(BaseAgent):
    """Случайные действия, без обучения."""

    def act(self, radar: RadarReading, state: Optional[dict] = None) -> int:
        self.controller_mode = ControllerMode.RANDOM
        return random.choice(ALL_ACTIONS)

    def learn_from_attempt(self, attempt: Attempt) -> None:
        pass


class RadarFoodAgent(BaseAgent):
    """Старая заглушка: всегда стоит."""

    def act(self, radar: RadarReading, state: Optional[dict] = None) -> int:
        return ACTION_STAY

    def learn_from_attempt(self, attempt: Attempt) -> None:
        pass


class NeuralFoodAgent(BaseAgent):
    """
    Нейросеть управляет движком; при плохой оценке — случайное блуждание.
    Опыт пишется всегда. Дообучение:
      — после еды в random (если хватает успешных шагов);
      — периодически после еды в neural (каждые TRAIN_EVERY_N_NEURAL_FOODS).
    После обучения — частичная очистка буфера и режим neural.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.network = FoodPolicyNetwork()
        self.evaluator = PerformanceEvaluator()
        self.controller_mode = ControllerMode.RANDOM
        self.food_total = 0
        self.random_foods = 0
        self.neural_foods_since_train = 0
        self.cleanup_count = 0
        self.last_cleanup_removed = 0
        self.last_train_samples = 0
        self._pending_train = False

    def reset(self) -> None:
        super().reset()
        self.evaluator.reset()
        self.controller_mode = ControllerMode.RANDOM
        self.food_total = 0
        self.random_foods = 0
        self.neural_foods_since_train = 0
        self.cleanup_count = 0
        self.last_cleanup_removed = 0
        self.last_train_samples = 0
        self._pending_train = False

    def act(self, radar: RadarReading, state: Optional[dict] = None) -> int:
        if self.controller_mode == ControllerMode.TRAINING:
            return random.choice(ALL_ACTIONS)

        if self.controller_mode == ControllerMode.NEURAL and self.network.is_trained:
            return self.network.predict_action(radar)

        self.controller_mode = ControllerMode.RANDOM
        return random.choice(ALL_ACTIONS)

    def learn_from_attempt(self, attempt: Attempt) -> None:
        if attempt.outcome == AttemptOutcome.SUCCESS:
            self.food_total += 1
            if attempt.source_mode == ControllerMode.RANDOM.value:
                self.random_foods += 1
                self._maybe_request_training(reason="random_food")
            elif attempt.source_mode == ControllerMode.NEURAL.value:
                self.neural_foods_since_train += 1
                if self.neural_foods_since_train >= TRAIN_EVERY_N_NEURAL_FOODS:
                    self._maybe_request_training(reason="neural_food")

        if attempt.source_mode == ControllerMode.NEURAL.value:
            self.evaluator.record_attempt(attempt)
            if self.evaluator.should_switch_to_random():
                self.controller_mode = ControllerMode.RANDOM

    def _success_step_count(self) -> int:
        return len(
            self.history.successful_steps_from_attempts(self.attempt_history.all())
        )

    def _maybe_request_training(self, reason: str = "") -> None:
        """Запросить обучение, если в буфере достаточно успешных шагов."""
        if self._success_step_count() >= TRAIN_MIN_SUCCESS_STEPS:
            self._pending_train = True

    def needs_training(self) -> bool:
        return self._pending_train

    def maybe_train(self) -> bool:
        """
        Если запрошено обучение — выполнить fit, почистить буфер, включить NN.
        Возвращает True, если обучение реально запускалось.
        """
        if not self._pending_train:
            return False

        self._pending_train = False
        train_steps = self.history.successful_steps_from_attempts(
            self.attempt_history.all()
        )
        if len(train_steps) < TRAIN_MIN_SUCCESS_STEPS:
            # данных мало — остаёмся в текущей логике act()
            if not self.network.is_trained:
                self.controller_mode = ControllerMode.RANDOM
            return False

        self.controller_mode = ControllerMode.TRAINING
        result = self.network.train_on_steps(train_steps)
        self.last_train_samples = result.get("samples", 0)

        removed_steps = self.history.keep_newest_fraction(EXPERIENCE_KEEP_FRACTION)
        removed_attempts = self.attempt_history.keep_newest_fraction(
            EXPERIENCE_KEEP_FRACTION
        )
        self.last_cleanup_removed = removed_steps + removed_attempts
        self.cleanup_count += 1

        self.evaluator.reset()
        self.random_foods = 0
        self.neural_foods_since_train = 0
        self.controller_mode = ControllerMode.NEURAL
        return True

    def dashboard_stats(self) -> dict:
        stats = super().dashboard_stats()
        eval_stats = self.evaluator.status_dict()
        loss = self.network.last_loss
        acc = self.network.last_accuracy
        stats.update(
            {
                "train_count": self.network.train_count,
                "last_train_samples": self.last_train_samples or "-",
                "last_loss": f"{loss:.3f}" if loss is not None else "-",
                "last_accuracy": f"{acc:.3f}" if acc is not None else "-",
                "cleanup_count": self.cleanup_count,
                "last_cleanup_removed": self.last_cleanup_removed,
                "food_total": self.food_total,
                "random_foods": self.random_foods,
                "neural_foods_since_train": self.neural_foods_since_train,
                "train_every_n_neural_foods": TRAIN_EVERY_N_NEURAL_FOODS,
                "eval_min_samples": EVAL_MIN_SAMPLES,
                **eval_stats,
                "eval_success_rate": (
                    f"{eval_stats['eval_success_rate']:.2f}"
                    if eval_stats["eval_success_rate"] is not None
                    else "-"
                ),
            }
        )
        return stats
