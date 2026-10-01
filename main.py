"""
Шаблон ИИ-агента: радар → опыт → диспетчер (random/NN) → движок.
"""

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import pygame

from config import CONTROL_AI, CONTROL_KEYBOARD
from game import Game

# Смените на CONTROL_KEYBOARD для ручного управления.
# CONTROL_AI запускает популяцию: несколько особей, отбор по времени жизни.
CONTROL_MODE = CONTROL_AI


def main():
    pygame.init()

    game = Game(control_mode=CONTROL_MODE)
    if CONTROL_MODE == CONTROL_KEYBOARD:
        game.run_keyboard()
    else:
        game.run_population()


if __name__ == "__main__":
    main()
