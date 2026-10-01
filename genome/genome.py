"""Корень компоновщика: геном ученика из четырёх блоков."""

import json
import random
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

from genome.blocks import (
    ArchitectureBlock,
    BehaviorBlock,
    GeneBlock,
    InnateWeightsBlock,
    LearningBlock,
)
from genome.spec import GENOME_VERSION


PathLike = Union[str, Path]


class Genome:
    """
    Компоновщик.

    Дети корня — блоки (архитектура, врождённые веса, поведение, обучение).
    Дети блока — гены. Мутация заходит в один блок и меняет один-два гена.
    Скрещивание обменивает блоки целиком.
    """

    def __init__(
        self,
        architecture: Optional[ArchitectureBlock] = None,
        innate_weights: Optional[InnateWeightsBlock] = None,
        behavior: Optional[BehaviorBlock] = None,
        learning: Optional[LearningBlock] = None,
    ):
        self._blocks = {
            "architecture": architecture if architecture is not None else ArchitectureBlock(),
            "innate_weights": innate_weights if innate_weights is not None else InnateWeightsBlock(),
            "behavior": behavior if behavior is not None else BehaviorBlock(),
            "learning": learning if learning is not None else LearningBlock(),
        }
        self._require_compatible_snapshot()

    @classmethod
    def default(cls) -> "Genome":
        """Стартовые значения из config.py, снимок весов пустой, inherit_weights = 0.5."""
        return cls()

    @classmethod
    def randomized(cls, rng: random.Random) -> "Genome":
        """Случайная особь: каждый ген внутри своей сетки, снимка весов ещё нет."""
        genome = cls()
        for block in genome.blocks():
            block.randomize(rng)
        genome.innate_weights.clear_snapshot()
        return genome

    @property
    def architecture(self) -> ArchitectureBlock:
        return self._blocks["architecture"]

    @property
    def innate_weights(self) -> InnateWeightsBlock:
        return self._blocks["innate_weights"]

    @property
    def behavior(self) -> BehaviorBlock:
        return self._blocks["behavior"]

    @property
    def learning(self) -> LearningBlock:
        return self._blocks["learning"]

    def blocks(self) -> List[GeneBlock]:
        return list(self._blocks.values())

    def block(self, name: str) -> GeneBlock:
        return self._blocks[name]

    def copy(self) -> "Genome":
        return Genome(
            self.architecture.copy(),
            self.innate_weights.copy(),
            self.behavior.copy(),
            self.learning.copy(),
        )

    def mutate(self, rng: random.Random, n_genes: int = 1) -> Tuple[str, List[str]]:
        """
        Один блок, внутри него n_genes генов (или все гены блока, если их меньше).
        Смена архитектуры, после которой снимок больше не подходит, очищает снимок.
        """
        if n_genes < 1:
            raise ValueError("n_genes должен быть >= 1")
        chosen = rng.choice(self.blocks())
        gene_names = chosen.mutate_n(rng, n_genes)
        if chosen.name == "architecture" and not self.innate_weights.matches(self.architecture):
            self.innate_weights.clear_snapshot()
        return chosen.name, gene_names

    def crossover(self, other: "Genome", rng: random.Random) -> "Genome":
        """
        Ребёнок получает каждый блок целиком от одного из родителей.
        Блок весов переносится только если снимок подходит архитектуре ребёнка.
        Неподходящий снимок сбрасывается: рождение будет случайным.
        """
        child_blocks = {}
        for name, block in self._blocks.items():
            if name == "innate_weights":
                continue
            donor = other if rng.random() < 0.5 else self
            child_blocks[name] = donor.block(name).copy()

        child_arch = child_blocks["architecture"]
        weights_donor = other if rng.random() < 0.5 else self
        weights = weights_donor.innate_weights.copy()
        if not weights.matches(child_arch):
            weights.clear_snapshot()
        child_blocks["innate_weights"] = weights
        return Genome(
            child_blocks["architecture"],
            child_blocks["innate_weights"],
            child_blocks["behavior"],
            child_blocks["learning"],
        )

    def store_birth_weights(self, arrays: Sequence[np.ndarray]) -> None:
        """
        Записать снимок на момент рождения, до обучения за эту жизнь.
        Обучение за жизнь этот метод не вызывает.
        """
        self.innate_weights.set_snapshot(arrays, self.architecture)

    def to_dict(self) -> dict:
        return {
            "version": GENOME_VERSION,
            "blocks": {name: block.to_dict() for name, block in self._blocks.items()},
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Genome":
        version = data.get("version")
        if version != GENOME_VERSION:
            raise ValueError(f"версия генома {version}, ожидалась {GENOME_VERSION}")
        blocks = data["blocks"]
        genome = cls(
            ArchitectureBlock.from_dict(blocks["architecture"]),
            InnateWeightsBlock.from_dict(blocks["innate_weights"]),
            BehaviorBlock.from_dict(blocks["behavior"]),
            LearningBlock.from_dict(blocks["learning"]),
        )
        return genome

    def save(self, path: PathLike) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: PathLike) -> "Genome":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data)

    def _require_compatible_snapshot(self) -> None:
        if not self.innate_weights.matches(self.architecture):
            raise ValueError("снимок весов не совпадает с архитектурой")
