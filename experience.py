"""Хранение истории: радар, действия агента, получение еды."""

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Iterable, List, Optional, Tuple


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
    food_gained — True, если на этом шаге агент съел зелёный квадрат.
    """

    step_index: int
    radar: RadarReading
    action: int
    food_gained: bool
    food_count: int = 0  # сколько еды съедено на шаге (обычно 0 или 1)


@dataclass
class ExperienceHistory:
    """История показаний радара, действий и фактов получения еды."""

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
    ) -> ExperienceStep:
        step = ExperienceStep(
            step_index=step_index,
            radar=radar_reading,
            action=action,
            food_gained=food_gained,
            food_count=food_count,
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
