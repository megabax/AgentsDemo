"""
Шаблон ИИ-агента: радар → опыт → движок; Keras-политика с random fallback.
"""

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import pygame

from config import CONTROL_AI, CONTROL_KEYBOARD
from agents import NeuralFoodAgent
from game import Game

# Смените на CONTROL_KEYBOARD для ручного управления
CONTROL_MODE = CONTROL_AI

# NeuralFoodAgent — Keras + random walk + переобучение
AGENT_CLASS = NeuralFoodAgent


def main():
    pygame.init()

    game = Game(control_mode=CONTROL_MODE)
    if CONTROL_MODE == CONTROL_KEYBOARD:
        game.run_keyboard()
    else:
        agent = AGENT_CLASS()
        game.run_with_ai(agent)


if __name__ == "__main__":
    main()
