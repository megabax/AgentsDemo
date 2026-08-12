"""Агенты ИИ: базовый интерфейс и заглушки (обучение пока не реализовано)."""

import random
from typing import Optional

from config import ATTEMPT_MAX_STEPS
from engine import ALL_ACTIONS, ACTION_STAY
from experience import (
    Attempt,
    AttemptHistory,
    AttemptOutcome,
    ExperienceHistory,
    ExperienceStep,
    RadarReading,
)


class BaseAgent:
    """
    Каркас агента: идёт к еде по радару через попытки (несколько шагов → исход).
    Покадровый опыт + журнал закрытых Attempt; learn — заглушка.
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

    def reset(self) -> None:
        self.history.clear()
        self.attempt_history.clear()
        self.step_index = 0
        self.attempt_index = 0
        self.current_attempt = None

    def choose_direction(self, radar: RadarReading) -> Optional[int]:
        """
        Заглушка: выбрать направление/гипотезу в начале попытки по радару.
        Например индекс луча с зелёным или код действия. Пока None.
        """
        return None

    def begin_attempt(self, radar: RadarReading) -> Attempt:
        """Открыть новую попытку дойти до еды."""
        if self.current_attempt is not None and self.current_attempt.is_open:
            self.finish_attempt(AttemptOutcome.ABORTED)

        attempt = Attempt(
            attempt_index=self.attempt_index,
            max_steps=self.max_steps_per_attempt,
            intended_direction=self.choose_direction(radar),
            initial_radar=radar,
        )
        self.attempt_index += 1
        self.current_attempt = attempt
        return attempt

    def act(self, radar: RadarReading, state: Optional[dict] = None) -> int:
        """
        Выбрать действие по радару (и опционально полному state).
        Должен вернуть код действия: 0↑ 1↓ 2← 3→ 4 стоять.
        """
        raise NotImplementedError

    def observe(
        self,
        radar: RadarReading,
        action: int,
        food_gained: bool,
        food_count: int = 0,
    ) -> ExperienceStep:
        """Записать шаг; при необходимости открыть/закрыть попытку."""
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
        """Закрыть текущую попытку, положить в журнал и дать learn-заглушке шанс."""
        if self.current_attempt is None or not self.current_attempt.is_open:
            return None

        attempt = self.current_attempt
        attempt.close(outcome)
        self.attempt_history.add(attempt)
        self.on_attempt_end(attempt)
        self.current_attempt = None
        return attempt

    def on_attempt_end(self, attempt: Attempt) -> None:
        """Хук: попытка завершена (успех/провал). Здесь будет обучение."""
        self.learn_from_attempt(attempt)

    def learn_from_attempt(self, attempt: Attempt) -> None:
        """Обучение по цепочке шагов и отложенному исходу. Пока заглушка."""
        pass

    def learn(self) -> None:
        """Покадровое обучение (если понадобится). Пока заглушка."""
        pass

    def on_episode_end(self) -> None:
        """Конец игровой сессии: закрыть висящую попытку и обучиться."""
        if self.current_attempt is not None and self.current_attempt.is_open:
            self.finish_attempt(AttemptOutcome.ABORTED)
        self.learn()


class DummyAgent(BaseAgent):
    """Заглушка: случайные действия, без обучения."""

    def act(self, radar: RadarReading, state: Optional[dict] = None) -> int:
        return random.choice(ALL_ACTIONS)

    def learn_from_attempt(self, attempt: Attempt) -> None:
        pass


class RadarFoodAgent(BaseAgent):
    """
    Заглушка будущего ИИ: учится идти к зелёным целям по радару
    на основе исхода попытки (дошёл / не дошёл за N шагов).
    Пока всегда стоит на месте.
    """

    def choose_direction(self, radar: RadarReading) -> Optional[int]:
        # TODO: выбрать луч/направление с зелёным сигналом
        return None

    def act(self, radar: RadarReading, state: Optional[dict] = None) -> int:
        # TODO: двигаться согласно intended_direction текущей попытки
        return ACTION_STAY

    def learn_from_attempt(self, attempt: Attempt) -> None:
        # TODO: усилить/ослабить гипотезу направления по attempt.outcome
        pass
