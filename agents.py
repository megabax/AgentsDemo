"""Агенты ИИ: базовый интерфейс и заглушки (обучение пока не реализовано)."""

import random
from typing import Optional

from engine import ALL_ACTIONS, ACTION_STAY
from experience import ExperienceHistory, ExperienceStep, RadarReading


class BaseAgent:
    """
    Каркас агента, который учится ходить к еде по радару.
    Сейчас только интерфейс и запись опыта; learn — заглушка.
    """

    def __init__(self, history: Optional[ExperienceHistory] = None):
        self.history = history if history is not None else ExperienceHistory()
        self.step_index = 0

    def reset(self) -> None:
        self.history.clear()
        self.step_index = 0

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
        """Записать шаг в историю опыта."""
        step = self.history.record(
            step_index=self.step_index,
            radar_reading=radar,
            action=action,
            food_gained=food_gained,
            food_count=food_count,
        )
        self.step_index += 1
        return step

    def learn(self) -> None:
        """Обучение по накопленной истории. Пока заглушка."""
        pass

    def on_episode_end(self) -> None:
        """Хук конца эпизода. Пока вызывает learn()."""
        self.learn()


class DummyAgent(BaseAgent):
    """Заглушка: случайные действия, без обучения."""

    def act(self, radar: RadarReading, state: Optional[dict] = None) -> int:
        return random.choice(ALL_ACTIONS)

    def learn(self) -> None:
        pass


class RadarFoodAgent(BaseAgent):
    """
    Заглушка будущего ИИ: будет учиться идти к зелёным целям по радару.
    Пока всегда стоит на месте.
    """

    def act(self, radar: RadarReading, state: Optional[dict] = None) -> int:
        # TODO: политика по показаниям радара (направление к зелёному)
        return ACTION_STAY

    def learn(self) -> None:
        # TODO: обучение на self.history (радар → действие → еда)
        pass
