"""Блоки генома — составные компоненты. Скрещивание обменивает блок целиком."""

import base64
import random
from typing import Dict, List, Optional, Sequence

import numpy as np

from config import (
    CYCLE_MIN_REPEATS,
    EXPERIENCE_KEEP_FRACTION,
    HISTORY_LEN,
    LABEL_STEP_PENALTY,
    NEURAL_STICKY_STEPS,
    NEURAL_SWITCH_MARGIN,
    NN_HIDDEN_1,
    NN_HIDDEN_2,
    PAIN_REPEAT_TO_FORBID,
    RADAR_RAY_COUNT,
    RANDOM_WALK_MAX_STEPS,
    RANDOM_WALK_MIN_STEPS,
    TRAIN_EVERY_N_FOODS,
)
from genome.genes import ChoiceGene, FloatGene, Gene, IntGene
from genome.spec import (
    CYCLE_REPEATS,
    HIDDEN_WIDTHS,
    HISTORY_LENGTHS,
    INHERIT_WEIGHTS,
    KEEP_FRACTION,
    LABEL_PENALTY,
    NUM_MOVEMENT_ACTIONS,
    PAIN_REPEAT,
    RANDOM_WALK_MAX_STEPS as WALK_MAX_RANGE,
    RANDOM_WALK_MIN_STEPS as WALK_MIN_RANGE,
    RAY_CHANNELS,
    STICKY_STEPS,
    SWITCH_MARGIN,
    TRAIN_EVERY,
)


def weight_shapes(hidden_1: int, hidden_2: int, history_len: int) -> tuple:
    """
    Формы матриц Keras Sequential: Dense(hidden_1), Dense(hidden_2), Dense(4).
    На каждый Dense — ядро, затем смещение. Порядок совпадает с model.get_weights().
    """
    frame = RADAR_RAY_COUNT * RAY_CHANNELS + NUM_MOVEMENT_ACTIONS
    n_in = history_len * frame
    return (
        (n_in, hidden_1),
        (hidden_1,),
        (hidden_1, hidden_2),
        (hidden_2,),
        (hidden_2, NUM_MOVEMENT_ACTIONS),
        (NUM_MOVEMENT_ACTIONS,),
    )


def encode_array(array: np.ndarray) -> dict:
    arr = np.ascontiguousarray(array, dtype=np.float32)
    return {
        "shape": list(arr.shape),
        "data": base64.b64encode(arr.tobytes()).decode("ascii"),
    }


def decode_array(payload: dict) -> np.ndarray:
    shape = tuple(int(item) for item in payload["shape"])
    raw = base64.b64decode(payload["data"])
    array = np.frombuffer(raw, dtype=np.float32).reshape(shape)
    return np.array(array, dtype=np.float32, copy=True)


class GeneBlock:
    """
    Составной компонент: набор листьев-генов.
    Мутация выбирает гены внутри блока. Скрещивание забирает блок целиком.
    """

    def __init__(self, name: str, genes: Sequence[Gene]):
        self.name = name
        self._genes: Dict[str, Gene] = {}
        for gene in genes:
            if gene.name in self._genes:
                raise ValueError(f"повтор гена {gene.name} в блоке {name}")
            self._genes[gene.name] = gene

    def gene(self, name: str) -> Gene:
        return self._genes[name]

    def gene_names(self) -> List[str]:
        return list(self._genes)

    def copy(self) -> "GeneBlock":
        raise NotImplementedError

    def mutate_n(self, rng: random.Random, n_genes: int) -> List[str]:
        if n_genes < 1:
            raise ValueError("n_genes должен быть >= 1")
        names = self.gene_names()
        chosen = rng.sample(names, min(n_genes, len(names)))
        for name in chosen:
            self._genes[name].mutate(rng)
        self.repair()
        return chosen

    def repair(self) -> None:
        """Поправить связи между генами после мутации."""

    def to_dict(self) -> dict:
        return {name: gene.to_dict() for name, gene in self._genes.items()}


class ArchitectureBlock(GeneBlock):
    """Ширина скрытых слоёв и длина истории. Только готовая сетка."""

    def __init__(
        self,
        hidden_1: int = NN_HIDDEN_1,
        hidden_2: int = NN_HIDDEN_2,
        history_len: int = HISTORY_LEN,
    ):
        super().__init__(
            "architecture",
            [
                ChoiceGene("hidden_1", hidden_1, HIDDEN_WIDTHS),
                ChoiceGene("hidden_2", hidden_2, HIDDEN_WIDTHS),
                ChoiceGene("history_len", history_len, HISTORY_LENGTHS),
            ],
        )

    @property
    def hidden_1(self) -> int:
        return self.gene("hidden_1").value

    @property
    def hidden_2(self) -> int:
        return self.gene("hidden_2").value

    @property
    def history_len(self) -> int:
        return self.gene("history_len").value

    def shapes(self) -> tuple:
        return weight_shapes(self.hidden_1, self.hidden_2, self.history_len)

    def same_as(self, other: "ArchitectureBlock") -> bool:
        return (
            self.hidden_1 == other.hidden_1
            and self.hidden_2 == other.hidden_2
            and self.history_len == other.history_len
        )

    def copy(self) -> "ArchitectureBlock":
        return ArchitectureBlock(self.hidden_1, self.hidden_2, self.history_len)

    @classmethod
    def from_dict(cls, data: dict) -> "ArchitectureBlock":
        return cls(data["hidden_1"], data["hidden_2"], data["history_len"])


class InnateWeightsBlock(GeneBlock):
    """
    Вероятность наследовать снимок и сам снимок.
    Снимок — состояние блока, не отдельный мутирующий ген:
    шум добавляется один раз при рождении, обучение за жизнь его не переписывает.
    """

    def __init__(self, inherit_weights: float = 0.5, snapshot: Optional[Sequence[np.ndarray]] = None):
        super().__init__(
            "innate_weights",
            [FloatGene("inherit_weights", inherit_weights, *INHERIT_WEIGHTS)],
        )
        self._snapshot: Optional[List[np.ndarray]] = None
        if snapshot is not None:
            self._snapshot = [np.array(array, dtype=np.float32, copy=True) for array in snapshot]

    @property
    def inherit_weights(self) -> float:
        return self.gene("inherit_weights").value

    @property
    def snapshot(self) -> Optional[List[np.ndarray]]:
        if self._snapshot is None:
            return None
        return [array.copy() for array in self._snapshot]

    def set_snapshot(self, arrays: Sequence[np.ndarray], architecture: ArchitectureBlock) -> None:
        expected = architecture.shapes()
        if len(arrays) != len(expected):
            raise ValueError("число матриц не совпадает с архитектурой")
        stored = []
        for array, shape in zip(arrays, expected):
            arr = np.array(array, dtype=np.float32, copy=True)
            if arr.shape != shape:
                raise ValueError(f"форма {arr.shape} вместо {shape}")
            stored.append(arr)
        self._snapshot = stored

    def clear_snapshot(self) -> None:
        self._snapshot = None

    def matches(self, architecture: ArchitectureBlock) -> bool:
        if self._snapshot is None:
            return True
        expected = architecture.shapes()
        if len(self._snapshot) != len(expected):
            return False
        return all(array.shape == shape for array, shape in zip(self._snapshot, expected))

    def copy(self) -> "InnateWeightsBlock":
        return InnateWeightsBlock(self.inherit_weights, self._snapshot)

    def to_dict(self) -> dict:
        data = super().to_dict()
        if self._snapshot is None:
            data["snapshot"] = None
        else:
            data["snapshot"] = [encode_array(array) for array in self._snapshot]
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "InnateWeightsBlock":
        raw = data["snapshot"]
        snapshot = None if raw is None else [decode_array(item) for item in raw]
        return cls(data["inherit_weights"], snapshot)


class BehaviorBlock(GeneBlock):
    """Характер хода: липкость, серия блуждания, боль, цикл."""

    def __init__(
        self,
        neural_sticky_steps: int = NEURAL_STICKY_STEPS,
        neural_switch_margin: float = NEURAL_SWITCH_MARGIN,
        random_walk_min_steps: int = RANDOM_WALK_MIN_STEPS,
        random_walk_max_steps: int = RANDOM_WALK_MAX_STEPS,
        pain_repeat_to_forbid: int = PAIN_REPEAT_TO_FORBID,
        cycle_min_repeats: int = CYCLE_MIN_REPEATS,
    ):
        super().__init__(
            "behavior",
            [
                IntGene("neural_sticky_steps", neural_sticky_steps, *STICKY_STEPS),
                FloatGene("neural_switch_margin", neural_switch_margin, *SWITCH_MARGIN),
                IntGene("random_walk_min_steps", random_walk_min_steps, *WALK_MIN_RANGE),
                IntGene("random_walk_max_steps", random_walk_max_steps, *WALK_MAX_RANGE),
                IntGene("pain_repeat_to_forbid", pain_repeat_to_forbid, *PAIN_REPEAT),
                IntGene("cycle_min_repeats", cycle_min_repeats, *CYCLE_REPEATS),
            ],
        )
        self.repair()

    def repair(self) -> None:
        low = self.gene("random_walk_min_steps")
        high = self.gene("random_walk_max_steps")
        if low.value > high.value:
            low.value, high.value = high.value, low.value

    def copy(self) -> "BehaviorBlock":
        return BehaviorBlock(
            self.gene("neural_sticky_steps").value,
            self.gene("neural_switch_margin").value,
            self.gene("random_walk_min_steps").value,
            self.gene("random_walk_max_steps").value,
            self.gene("pain_repeat_to_forbid").value,
            self.gene("cycle_min_repeats").value,
        )

    @classmethod
    def from_dict(cls, data: dict) -> "BehaviorBlock":
        return cls(
            data["neural_sticky_steps"],
            data["neural_switch_margin"],
            data["random_walk_min_steps"],
            data["random_walk_max_steps"],
            data["pain_repeat_to_forbid"],
            data["cycle_min_repeats"],
        )


class LearningBlock(GeneBlock):
    """Насколько жадно ученик верит недавнему опыту."""

    def __init__(
        self,
        label_step_penalty: float = LABEL_STEP_PENALTY,
        train_every_n_foods: int = TRAIN_EVERY_N_FOODS,
        experience_keep_fraction: float = EXPERIENCE_KEEP_FRACTION,
    ):
        super().__init__(
            "learning",
            [
                FloatGene("label_step_penalty", label_step_penalty, *LABEL_PENALTY),
                IntGene("train_every_n_foods", train_every_n_foods, *TRAIN_EVERY),
                FloatGene("experience_keep_fraction", experience_keep_fraction, *KEEP_FRACTION),
            ],
        )

    def copy(self) -> "LearningBlock":
        return LearningBlock(
            self.gene("label_step_penalty").value,
            self.gene("train_every_n_foods").value,
            self.gene("experience_keep_fraction").value,
        )

    @classmethod
    def from_dict(cls, data: dict) -> "LearningBlock":
        return cls(
            data["label_step_penalty"],
            data["train_every_n_foods"],
            data["experience_keep_fraction"],
        )
