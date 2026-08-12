"""
Простой шаблон проекта для ИИ-агента в компьютерной игре.
Игра: управление красным квадратом, который должен собирать зелёные цели.
Инфраструктура ИИ: радар → история опыта → движок движения (обучение пока заглушка).
"""

import pygame

from config import CONTROL_AI, CONTROL_KEYBOARD
from agents import DummyAgent, RadarFoodAgent
from game import Game

# Смените на CONTROL_KEYBOARD для ручного управления
CONTROL_MODE = CONTROL_AI

# DummyAgent — случайные ходы; RadarFoodAgent — заглушка будущего ИИ по радару
AGENT_CLASS = RadarFoodAgent #DummyAgent


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
