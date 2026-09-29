"""Рождение весов: ген inherit_weights выбирает копию снимка или случайный старт."""

import random
from typing import List, Optional

import numpy as np

from genome.spec import BIRTH_WEIGHT_NOISE_STD


SOURCE_INHERIT = "inherit"
SOURCE_RANDOM = "random"


class BirthDecision:
    """Чем инициализировать сеть в момент рождения, до обучения за жизнь."""

    def __init__(self, source: str, weights: Optional[List[np.ndarray]]):
        if source not in (SOURCE_INHERIT, SOURCE_RANDOM):
            raise ValueError(f"неизвестный источник рождения: {source}")
        self.source = source
        self.weights = weights

    @property
    def inherited(self) -> bool:
        return self.source == SOURCE_INHERIT


def decide_birth(genome, rng: random.Random, noise_std: float = BIRTH_WEIGHT_NOISE_STD) -> BirthDecision:
    """
    Жребий рождения.

    Наследование — только если снимок есть и его формы совпали с архитектурой детёныша.
    Иначе всегда случайная инициализация, каким бы ни был inherit_weights.
    При наследовании к копии добавляется слабый шум, сам геном здесь не меняется.
    """
    innate = genome.innate_weights
    snapshot = innate.snapshot
    if snapshot is None or not innate.matches(genome.architecture):
        return BirthDecision(SOURCE_RANDOM, None)
    if rng.random() >= innate.inherit_weights:
        return BirthDecision(SOURCE_RANDOM, None)

    noise_rng = np.random.default_rng(rng.randrange(2**32))
    weights = []
    for array in snapshot:
        noise = noise_rng.normal(0.0, noise_std, size=array.shape).astype(np.float32)
        weights.append(array + noise)
    return BirthDecision(SOURCE_INHERIT, weights)
