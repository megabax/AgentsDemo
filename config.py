"""Общие константы игры."""

import os

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PLAYER_SPRITE_PATH = os.path.join(_BASE_DIR, "images", "cow.png")

WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
FPS = 60

WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
BLACK = (0, 0, 0)

PLAYER_SIZE = 40
PLAYER_SPEED = 5

TARGET_SIZE = 30
TARGET_COUNT = 5
