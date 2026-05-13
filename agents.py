"""Агент-заглушка: случайные действия вместо настоящего ИИ."""

import random


class DummyAgent:
    def __init__(self):
        self.actions = [0, 1, 2, 3, 4]  # вверх, вниз, влево, вправо, стоять

    def act(self, state):
        return random.choice(self.actions)

    def learn(self, state, action, reward, next_state, done):
        pass
