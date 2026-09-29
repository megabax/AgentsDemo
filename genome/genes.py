"""Листья компоновщика: один ген — одно число из заданной сетки или диапазона."""

import random
from abc import ABC, abstractmethod
from typing import Sequence, Tuple


class Gene(ABC):
    """Компонент-лист. Мутация меняет только его значение."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def copy(self) -> "Gene":
        raise NotImplementedError

    @abstractmethod
    def mutate(self, rng: random.Random) -> None:
        raise NotImplementedError

    @abstractmethod
    def to_dict(self):
        raise NotImplementedError


class ChoiceGene(Gene):
    """Значение только из фиксированного кортежа. Мутация сдвигает на соседний вариант."""

    def __init__(self, name: str, value: int, options: Sequence[int]):
        super().__init__(name)
        self.options: Tuple[int, ...] = tuple(int(item) for item in options)
        if not self.options:
            raise ValueError(f"{name}: пустая сетка значений")
        if int(value) not in self.options:
            raise ValueError(f"{name}={value} вне сетки {self.options}")
        self.value = int(value)

    def copy(self) -> "ChoiceGene":
        return ChoiceGene(self.name, self.value, self.options)

    def mutate(self, rng: random.Random) -> None:
        index = self.options.index(self.value)
        if index == 0:
            self.value = self.options[1] if len(self.options) > 1 else self.value
        elif index == len(self.options) - 1:
            self.value = self.options[index - 1]
        else:
            self.value = self.options[index + rng.choice((-1, 1))]

    def to_dict(self) -> int:
        return self.value


class IntGene(Gene):
    """Целое в диапазоне [low, high]. Мутация — шаг ±1."""

    def __init__(self, name: str, value: int, low: int, high: int):
        super().__init__(name)
        self.low = int(low)
        self.high = int(high)
        if self.low > self.high:
            raise ValueError(f"{name}: low > high")
        self.value = self._clamp(int(value))

    def _clamp(self, value: int) -> int:
        if value < self.low or value > self.high:
            raise ValueError(f"{self.name}={value} вне [{self.low}, {self.high}]")
        return value

    def copy(self) -> "IntGene":
        return IntGene(self.name, self.value, self.low, self.high)

    def mutate(self, rng: random.Random) -> None:
        nudged = self.value + rng.choice((-1, 1))
        self.value = min(self.high, max(self.low, nudged))

    def to_dict(self) -> int:
        return self.value


class FloatGene(Gene):
    """Вещественное в диапазоне [low, high]. Мутация — один шаг step."""

    def __init__(self, name: str, value: float, low: float, high: float, step: float):
        super().__init__(name)
        self.low = float(low)
        self.high = float(high)
        self.step = float(step)
        if self.low > self.high or self.step <= 0:
            raise ValueError(f"{name}: некорректный диапазон или шаг")
        self.value = self._clamp(float(value))

    def _clamp(self, value: float) -> float:
        if value < self.low - 1e-9 or value > self.high + 1e-9:
            raise ValueError(f"{self.name}={value} вне [{self.low}, {self.high}]")
        return round(min(self.high, max(self.low, value)), 5)

    def copy(self) -> "FloatGene":
        return FloatGene(self.name, self.value, self.low, self.high, self.step)

    def mutate(self, rng: random.Random) -> None:
        nudged = self.value + rng.choice((-1.0, 1.0)) * self.step
        self.value = round(min(self.high, max(self.low, nudged)), 5)

    def to_dict(self) -> float:
        return self.value
