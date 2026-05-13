"""
Простой шаблон проекта для ИИ-агента в компьютерной игре.
Игра: управление спрайтом коровы, который должен собирать зелёные цели.
Агент-заглушка: просто двигается случайным образом.
"""

import pygame

from config import CONTROL_AI, CONTROL_KEYBOARD
from dummy_agent import DummyAgent
from game import Game

# Смените на CONTROL_AI для управления через класс Player и агента-заглушку
CONTROL_MODE = CONTROL_AI #CONTROL_KEYBOARD


def main():
    pygame.init()

    game = Game(control_mode=CONTROL_MODE)
    if CONTROL_MODE == CONTROL_KEYBOARD:
        game.run_keyboard()
    else:
        dummy_ai = DummyAgent()
        game.run_with_ai(dummy_ai)


if __name__ == "__main__":
    main()
