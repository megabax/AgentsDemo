"""
Простой шаблон проекта для ИИ-агента в компьютерной игре.
Игра: управление красным квадратом, который должен собирать зелёные цели.
Агент-заглушка: просто двигается случайным образом.
"""

import pygame

from dummy_agent import DummyAgent
from game import Game


def main():
    pygame.init()

    game = Game()
    dummy_ai = DummyAgent()
    game.run_with_ai(dummy_ai)


if __name__ == "__main__":
    main()
