"""Агенты ИИ: опыт, попытки; NeuralFoodAgent + диспетчер режимов."""

import random
from typing import Optional

from config import (
    ATTEMPT_MAX_STEPS,
    EXPERIENCE_KEEP_FRACTION,
    HISTORY_LEN,
    TRAIN_EVERY_N_FOODS,
    TRAIN_MIN_SAMPLES,
)
from dispatcher import ControllerMode, ModeDispatcher
from engine import ALL_ACTIONS, ACTION_STAY
from experience import (
    Attempt,
    AttemptHistory,
    AttemptOutcome,
    ExperienceHistory,
    ExperienceStep,
    RadarReading,
)
from nn_model import FoodPolicyNetwork, attempts_to_dataset, history_features_from_steps


class BaseAgent:
    """Каркас: радар → попытки → опыт."""

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
    def act(self, radar: RadarReading, state: Optional[dict] = None) -> int:
        self.controller_mode = ControllerMode.RANDOM
        return random.choice(ALL_ACTIONS)


class RadarFoodAgent(BaseAgent):
    def act(self, radar: RadarReading, state: Optional[dict] = None) -> int:
        return ACTION_STAY


class NeuralFoodAgent(BaseAgent):
    """
    Движком управляет ModeDispatcher (random ↔ neural).
    Опыт пишется всегда. Обучение на +/− попытках с историей HISTORY_LEN шагов.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.network = FoodPolicyNetwork()
        self.dispatcher = ModeDispatcher(self.network)
        self.controller_mode = self.dispatcher.mode
        self.food_total = 0
        self.foods_since_train = 0
        self.cleanup_count = 0
        self.last_cleanup_removed = 0
        self.last_train_samples = 0
        self.last_switch_reason = "-"
        self._pending_train = False
        self._cached_samples = 0

    def reset(self) -> None:
        super().reset()
        self.dispatcher.reset()
        self.controller_mode = self.dispatcher.mode
        self.food_total = 0
        self.foods_since_train = 0
        self.cleanup_count = 0
        self.last_cleanup_removed = 0
        self.last_train_samples = 0
        self.last_switch_reason = "-"
        self._pending_train = False
        self._cached_samples = 0

    def _sync_mode(self) -> None:
        self.controller_mode = self.dispatcher.mode

    def act(self, radar: RadarReading, state: Optional[dict] = None) -> int:
        past = self.history.recent(HISTORY_LEN - 1)
        features = history_features_from_steps(past, radar, HISTORY_LEN)
        action = self.dispatcher.choose_action(features)
        self._sync_mode()
        return action

    def learn_from_attempt(self, attempt: Attempt) -> None:
        switch = self.dispatcher.on_attempt_end(attempt)
        if switch:
            self.last_switch_reason = switch
        self._sync_mode()
        self._cached_samples = self._sample_count()

        if attempt.outcome == AttemptOutcome.SUCCESS:
            self.food_total += 1
            self.foods_since_train += 1
            if self.foods_since_train >= TRAIN_EVERY_N_FOODS:
                self._maybe_request_training()

        if attempt.outcome == AttemptOutcome.FAILURE:
            if self._cached_samples >= TRAIN_MIN_SAMPLES and self.foods_since_train > 0:
                self._maybe_request_training()

    def _sample_count(self) -> int:
        x, _ = attempts_to_dataset(self.attempt_history.all())
        return len(x)

    def _maybe_request_training(self) -> None:
        if self._cached_samples >= TRAIN_MIN_SAMPLES:
            self._pending_train = True

    def needs_training(self) -> bool:
        return self._pending_train

    def maybe_train(self) -> bool:
        if not self._pending_train:
            return False

        self._pending_train = False
        attempts = self.attempt_history.all()
        x, _ = attempts_to_dataset(attempts)
        if len(x) < TRAIN_MIN_SAMPLES:
            if not self.network.is_trained:
                self.dispatcher.set_random()
            self._sync_mode()
            return False

        self.dispatcher.set_training()
        self._sync_mode()
        result = self.network.train_on_attempts(attempts)
        self.last_train_samples = result.get("samples", 0)

        removed_steps = self.history.keep_newest_fraction(EXPERIENCE_KEEP_FRACTION)
        removed_attempts = self.attempt_history.keep_newest_fraction(
            EXPERIENCE_KEEP_FRACTION
        )
        self.last_cleanup_removed = removed_steps + removed_attempts
        self.cleanup_count += 1
        self.foods_since_train = 0

        self.dispatcher.set_neural()
        self._sync_mode()
        self._cached_samples = self._sample_count()
        return True

    def dashboard_stats(self) -> dict:
        stats = super().dashboard_stats()
        loss = self.network.last_loss
        acc = self.network.last_accuracy
        d = self.dispatcher.status_dict()
        stats.update(
            {
                **d,
                "train_count": self.network.train_count,
                "last_train_samples": self.last_train_samples or "-",
                "last_loss": f"{loss:.3f}" if loss is not None else "-",
                "last_accuracy": f"{acc:.3f}" if acc is not None else "-",
                "cleanup_count": self.cleanup_count,
                "last_cleanup_removed": self.last_cleanup_removed,
                "food_total": self.food_total,
                "foods_since_train": self.foods_since_train,
                "train_every_n_foods": TRAIN_EVERY_N_FOODS,
                "history_len": HISTORY_LEN,
                "last_switch_reason": self.last_switch_reason,
                "trainable_samples": self._cached_samples,
            }
        )
        return stats
