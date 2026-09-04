"""Keras-модель: история радара+действий → действие (+/− примеры)."""

from typing import List, Optional, Sequence, Tuple

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

from config import (
    HISTORY_LEN,
    NN_BATCH_SIZE,
    NN_EPOCHS,
    NN_HIDDEN_1,
    NN_HIDDEN_2,
    NN_LEARNING_RATE,
    RADAR_MAX_RANGE,
    RADAR_RAY_COUNT,
)
from engine import MOVEMENT_ACTIONS
from experience import Attempt, AttemptOutcome, ExperienceStep, RadarReading


NUM_ACTIONS = len(MOVEMENT_ACTIONS)
RADAR_FEAT_SIZE = RADAR_RAY_COUNT * 4  # distance + R + G + B
FRAME_FEAT_SIZE = RADAR_FEAT_SIZE + NUM_ACTIONS  # радар + one-hot действия
FEATURE_SIZE = HISTORY_LEN * FRAME_FEAT_SIZE


def radar_to_features(radar: RadarReading) -> np.ndarray:
    feats = np.empty(RADAR_FEAT_SIZE, dtype=np.float32)
    i = 0
    max_range = float(RADAR_MAX_RANGE) or 1.0
    for dist, color in zip(radar.distances, radar.colors):
        feats[i] = float(dist) / max_range
        feats[i + 1] = color[0] / 255.0
        feats[i + 2] = color[1] / 255.0
        feats[i + 3] = color[2] / 255.0
        i += 4
    return feats


def action_to_one_hot(action: Optional[int]) -> np.ndarray:
    """One-hot только по MOVEMENT_ACTIONS; STAY и прочее → нули."""
    vec = np.zeros(NUM_ACTIONS, dtype=np.float32)
    if action is not None and action in MOVEMENT_ACTIONS:
        vec[MOVEMENT_ACTIONS.index(action)] = 1.0
    return vec


def empty_radar_features() -> np.ndarray:
    return np.zeros(RADAR_FEAT_SIZE, dtype=np.float32)


def frame_features(radar: Optional[RadarReading], action: Optional[int]) -> np.ndarray:
    """Один кадр: радар + one-hot действия (action=None → нули)."""
    radar_f = radar_to_features(radar) if radar is not None else empty_radar_features()
    return np.concatenate([radar_f, action_to_one_hot(action)])


def history_features_from_steps(
    past_steps: Sequence[ExperienceStep],
    current_radar: RadarReading,
    history_len: int = HISTORY_LEN,
) -> np.ndarray:
    """
    Вектор для выбора действия сейчас:
    до (history_len-1) прошлых шагов (радар+действие) + текущий радар без действия.
    """
    frames: List[np.ndarray] = []
    # прошлые кадры (старые → новые), нужны history_len-1 штук
    need_past = history_len - 1
    past = list(past_steps)[-need_past:] if need_past > 0 else []
    pad = need_past - len(past)
    for _ in range(pad):
        frames.append(frame_features(None, None))
    for step in past:
        frames.append(frame_features(step.radar, step.action))
    # текущий кадр: только радар (действие ещё не выбрано)
    frames.append(frame_features(current_radar, None))
    return np.concatenate(frames).astype(np.float32)


def history_features_at_index(
    steps: Sequence[ExperienceStep],
    index: int,
    history_len: int = HISTORY_LEN,
) -> np.ndarray:
    """Признаки для шага index внутри попытки (для обучения)."""
    past = steps[max(0, index - (history_len - 1)) : index]
    return history_features_from_steps(past, steps[index].radar, history_len)


def label_for_action(action: int, positive: bool) -> np.ndarray:
    """
    Положительный пример — one-hot направления.
    Отрицательный — равномерно по остальным направлениям (без STAY).
    Шаги со STAY в истории пропускаются на уровне attempts_to_dataset.
    """
    y = np.zeros(NUM_ACTIONS, dtype=np.float32)
    if action not in MOVEMENT_ACTIONS:
        # на всякий случай: не усиливать «стоять»
        return y
    idx = MOVEMENT_ACTIONS.index(action)
    if positive:
        y[idx] = 1.0
    else:
        share = 1.0 / max(1, NUM_ACTIONS - 1)
        for a in range(NUM_ACTIONS):
            if a != idx:
                y[a] = share
    return y


def attempts_to_dataset(
    attempts: Sequence[Attempt],
    history_len: int = HISTORY_LEN,
) -> Tuple[np.ndarray, np.ndarray]:
    """Сэмплы из успешных (+) и провальных (−) попыток; только шаги с движением."""
    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    for attempt in attempts:
        if attempt.outcome == AttemptOutcome.SUCCESS:
            positive = True
        elif attempt.outcome == AttemptOutcome.FAILURE:
            positive = False
        else:
            continue
        steps = attempt.steps
        for i in range(len(steps)):
            if steps[i].action not in MOVEMENT_ACTIONS:
                continue
            xs.append(history_features_at_index(steps, i, history_len))
            ys.append(label_for_action(steps[i].action, positive))

    if not xs:
        return (
            np.zeros((0, FEATURE_SIZE), dtype=np.float32),
            np.zeros((0, NUM_ACTIONS), dtype=np.float32),
        )
    return np.stack(xs), np.stack(ys)


def build_policy_model(input_size: int = FEATURE_SIZE) -> keras.Model:
    model = keras.Sequential(
        [
            layers.Input(shape=(input_size,)),
            layers.Dense(NN_HIDDEN_1, activation="relu"),
            layers.Dense(NN_HIDDEN_2, activation="relu"),
            layers.Dense(NUM_ACTIONS, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=NN_LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


class FoodPolicyNetwork:
    """Обёртка над Keras: предсказание по истории и обучение на +/−."""

    def __init__(self):
        self.model = build_policy_model()
        self.train_count = 0
        self.last_loss = None
        self.last_accuracy = None
        self.is_trained = False

    def predict_probs(self, features: np.ndarray) -> np.ndarray:
        x = features.astype(np.float32)[None, ...]
        return self.model(x, training=False).numpy()[0]

    def predict_action(self, features: np.ndarray) -> int:
        probs = self.predict_probs(features)
        idx = int(np.argmax(probs))
        return MOVEMENT_ACTIONS[idx]

    def train_on_attempts(self, attempts: Sequence[Attempt]) -> dict:
        x, y = attempts_to_dataset(attempts)
        if len(x) == 0:
            return {"samples": 0, "loss": None, "accuracy": None}

        history = self.model.fit(
            x,
            y,
            epochs=NN_EPOCHS,
            batch_size=min(NN_BATCH_SIZE, len(x)),
            verbose=0,
        )
        self.train_count += 1
        self.is_trained = True
        self.last_loss = float(history.history["loss"][-1])
        self.last_accuracy = float(history.history["accuracy"][-1])
        return {
            "samples": int(len(x)),
            "loss": self.last_loss,
            "accuracy": self.last_accuracy,
        }
