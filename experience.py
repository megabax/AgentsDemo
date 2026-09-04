"""Хранение истории: радар, действия, еда и попытки (эпизоды) с отложенным исходом."""

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Deque, Iterable, List, Optional, Tuple

from config import ATTEMPT_MAX_STEPS


Color = Tuple[int, int, int]


@dataclass(frozen=True)
class RadarReading:
    """Снимок радара на один шаг."""

    distances: Tuple[float, ...]
    colors: Tuple[Color, ...]

    @classmethod
    def from_radar(cls, radar) -> "RadarReading":
        return cls(
            distances=tuple(radar.distances),
            colors=tuple(tuple(c) for c in radar.colors),
        )

    @classmethod
    def from_dict(cls, data: dict) -> "RadarReading":
        return cls(
            distances=tuple(data["distances"]),
            colors=tuple(tuple(c) for c in data["colors"]),
        )


@dataclass(frozen=True)
class ExperienceStep:
    """
    Один шаг опыта для будущего обучения.
    food_gained — съел зелёный квадрат;
    pain — удар о стену (хотел идти, но упёрся).
    """

    step_index: int
    radar: RadarReading
    action: int
    food_gained: bool
    food_count: int = 0
    pain: bool = False


class AttemptOutcome(Enum):
    """Итог попытки дойти до еды за несколько шагов."""

    SUCCESS = "success"  # дошёл до еды
    FAILURE = "failure"  # не дошёл (таймаут / отказ)
    ABORTED = "aborted"  # попытка прервана снаружи
    PENDING = "pending"  # ещё идёт


@dataclass
class Attempt:
    """
    Попытка: серия шагов от решения «куда идти» до исхода.
    Обучение смотрит на attempt целиком, а не на один кадр.
    """

    attempt_index: int
    max_steps: int = ATTEMPT_MAX_STEPS
    # Гипотеза направления по радару в начале попытки (заглушка: индекс луча / код действия)
    intended_direction: Optional[int] = None
    initial_radar: Optional[RadarReading] = None
    steps: List[ExperienceStep] = field(default_factory=list)
    # Режим контроллера на момент попытки: "neural" / "random" / …
    source_mode: Optional[str] = None
    outcome: AttemptOutcome = AttemptOutcome.PENDING

    @property
    def is_open(self) -> bool:
        return self.outcome == AttemptOutcome.PENDING

    @property
    def length(self) -> int:
        return len(self.steps)

    def add_step(self, step: ExperienceStep) -> None:
        if not self.is_open:
            raise RuntimeError("Нельзя добавлять шаги в закрытую попытку")
        self.steps.append(step)

    def close(self, outcome: AttemptOutcome) -> None:
        if outcome == AttemptOutcome.PENDING:
            raise ValueError("Исход PENDING недопустим при закрытии")
        if not self.is_open:
            raise RuntimeError("Попытка уже закрыта")
        self.outcome = outcome

    def success(self) -> bool:
        return self.outcome == AttemptOutcome.SUCCESS


@dataclass
class AttemptHistory:
    """Журнал закрытых попыток (для обучения с отложенным исходом)."""

    max_size: Optional[int] = None
    _attempts: Deque[Attempt] = field(default_factory=deque, init=False)

    def __post_init__(self):
        if self.max_size is not None:
            self._attempts = deque(maxlen=self.max_size)

    def __len__(self) -> int:
        return len(self._attempts)

    def clear(self) -> None:
        self._attempts.clear()

    def add(self, attempt: Attempt) -> None:
        if attempt.is_open:
            raise ValueError("В историю кладут только закрытые попытки")
        self._attempts.append(attempt)

    def all(self) -> List[Attempt]:
        return list(self._attempts)

    def recent(self, n: int) -> List[Attempt]:
        if n <= 0:
            return []
        return list(self._attempts)[-n:]

    def successes(self) -> List[Attempt]:
        return [a for a in self._attempts if a.outcome == AttemptOutcome.SUCCESS]

    def failures(self) -> List[Attempt]:
        return [a for a in self._attempts if a.outcome == AttemptOutcome.FAILURE]

    def drop_oldest_fraction(self, drop_fraction: float) -> int:
        """Удалить старые попытки; вернуть число удалённых."""
        if not self._attempts or drop_fraction <= 0:
            return 0
        drop_fraction = min(1.0, drop_fraction)
        n = int(len(self._attempts) * drop_fraction)
        for _ in range(n):
            self._attempts.popleft()
        return n

    def keep_newest_fraction(self, keep_fraction: float) -> int:
        """Оставить только хвост; вернуть число удалённых."""
        keep_fraction = max(0.0, min(1.0, keep_fraction))
        return self.drop_oldest_fraction(1.0 - keep_fraction)


@dataclass
class ExperienceHistory:
    """История показаний радара, действий и фактов получения еды (покадровая)."""

    max_size: Optional[int] = None
    _steps: Deque[ExperienceStep] = field(default_factory=deque, init=False)

    def __post_init__(self):
        if self.max_size is not None:
            self._steps = deque(maxlen=self.max_size)

    def __len__(self) -> int:
        return len(self._steps)

    def clear(self) -> None:
        self._steps.clear()

    def record(
        self,
        step_index: int,
        radar_reading: RadarReading,
        action: int,
        food_gained: bool,
        food_count: int = 0,
        pain: bool = False,
    ) -> ExperienceStep:
        step = ExperienceStep(
            step_index=step_index,
            radar=radar_reading,
            action=action,
            food_gained=food_gained,
            food_count=food_count,
            pain=pain,
        )
        self._steps.append(step)
        return step

    def steps(self) -> List[ExperienceStep]:
        return list(self._steps)

    def recent(self, n: int) -> List[ExperienceStep]:
        if n <= 0:
            return []
        return list(self._steps)[-n:]

    def actions(self) -> List[int]:
        return [s.action for s in self._steps]

    def food_events(self) -> List[ExperienceStep]:
        return [s for s in self._steps if s.food_gained]

    def radar_readings(self) -> List[RadarReading]:
        return [s.radar for s in self._steps]

    def extend(self, steps: Iterable[ExperienceStep]) -> None:
        for step in steps:
            self._steps.append(step)

    def drop_oldest(self, count: int) -> int:
        """Удалить count старых записей; вернуть сколько реально удалили."""
        removed = 0
        count = max(0, count)
        while removed < count and self._steps:
            self._steps.popleft()
            removed += 1
        return removed

    def drop_oldest_fraction(self, drop_fraction: float) -> int:
        if not self._steps or drop_fraction <= 0:
            return 0
        drop_fraction = min(1.0, drop_fraction)
        return self.drop_oldest(int(len(self._steps) * drop_fraction))

    def keep_newest_fraction(self, keep_fraction: float) -> int:
        keep_fraction = max(0.0, min(1.0, keep_fraction))
        return self.drop_oldest_fraction(1.0 - keep_fraction)

    def successful_steps_from_attempts(self, attempts: Iterable[Attempt]) -> List[ExperienceStep]:
        """Шаги из успешных попыток — материал для обучения."""
        result: List[ExperienceStep] = []
        for attempt in attempts:
            if attempt.outcome == AttemptOutcome.SUCCESS:
                result.extend(attempt.steps)
        return result
