"""Keras-модель: радар → действие."""

from typing import Sequence, Tuple

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

from config import (
    NN_BATCH_SIZE,
    NN_EPOCHS,
    NN_HIDDEN_1,
    NN_HIDDEN_2,
    NN_LEARNING_RATE,
    RADAR_MAX_RANGE,
    RADAR_RAY_COUNT,
)
from engine import ALL_ACTIONS
from experience import ExperienceStep, RadarReading


NUM_ACTIONS = len(ALL_ACTIONS)
FEATURE_SIZE = RADAR_RAY_COUNT * 4  # distance + R + G + B per ray


def radar_to_features(radar: RadarReading) -> np.ndarray:
    """Вектор признаков радара shape (FEATURE_SIZE,)."""
    feats = np.empty(FEATURE_SIZE, dtype=np.float32)
    i = 0
    max_range = float(RADAR_MAX_RANGE) or 1.0
    for dist, color in zip(radar.distances, radar.colors):
        feats[i] = float(dist) / max_range
        feats[i + 1] = color[0] / 255.0
        feats[i + 2] = color[1] / 255.0
        feats[i + 3] = color[2] / 255.0
        i += 4
    return feats


def steps_to_dataset(steps: Sequence[ExperienceStep]) -> Tuple[np.ndarray, np.ndarray]:
    """X — радары, y — one-hot действий."""
    if not steps:
        return (
            np.zeros((0, FEATURE_SIZE), dtype=np.float32),
            np.zeros((0, NUM_ACTIONS), dtype=np.float32),
        )
    x = np.stack([radar_to_features(s.radar) for s in steps])
    y = keras.utils.to_categorical(
        [s.action for s in steps],
        num_classes=NUM_ACTIONS,
    )
    return x, y


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
    """Обёртка над Keras: предсказание действия и периодическое обучение."""

    def __init__(self):
        self.model = build_policy_model()
        self.train_count = 0
        self.last_loss = None
        self.last_accuracy = None
        self.is_trained = False

    def predict_action(self, radar: RadarReading) -> int:
        x = radar_to_features(radar)[None, ...]
        probs = self.model(x, training=False).numpy()[0]
        return int(np.argmax(probs))

    def train_on_steps(self, steps: Sequence[ExperienceStep]) -> dict:
        x, y = steps_to_dataset(steps)
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
