"""Игрок: базовый класс и вариант, управляемый ИИ-агентом."""

import pygame

from config import (
    BOUNDARY_MODE,
    BOUNDARY_MODE_WRAP,
    PLAYER_SIZE,
    PLAYER_SPEED,
    RED,
    WINDOW_HEIGHT,
    WINDOW_WIDTH,
)
from radar import Radar


class BasePlayer:
    """Общее состояние, отрисовка, границы, радар и снимок state для агента."""

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.size = PLAYER_SIZE
        self.rect = pygame.Rect(x, y, self.size, self.size)
        self.score = 0
        self.radar = Radar()

    def _apply_bounds(self):
        if BOUNDARY_MODE == BOUNDARY_MODE_WRAP:
            if self.x < 0:
                self.x = WINDOW_WIDTH - self.size
            elif self.x > WINDOW_WIDTH - self.size:
                self.x = 0
            if self.y < 0:
                self.y = WINDOW_HEIGHT - self.size
            elif self.y > WINDOW_HEIGHT - self.size:
                self.y = 0
        else:
            # BOUNCE без скорости = останов у края
            self.x = max(0, min(WINDOW_WIDTH - self.size, self.x))
            self.y = max(0, min(WINDOW_HEIGHT - self.size, self.y))

        self.rect.topleft = (self.x, self.y)

    def scan_radar(self, targets):
        """Обновить показания радара из центра игрока."""
        cx = self.x + self.size / 2
        cy = self.y + self.size / 2
        self.radar.scan(cx, cy, targets)

    def draw(self, screen):
        pygame.draw.rect(screen, RED, self.rect)

    def get_state(self, targets):
        """
        Получение текущего состояния для ИИ-агента.
        Возвращает: позицию игрока, позиции целей, очки и снимок радара.
        """
        return {
            "player_pos": (self.x, self.y),
            "targets_pos": [(t.x, t.y) for t in targets],
            "score": self.score,
            "radar": self.radar.as_dict(),
        }


class Player(BasePlayer):
    def update(self, action):
        """
        Обновление позиции. Возвращает True, если удар о стену
        (хотели сдвинуться, но позиция не изменилась — «боль»).
        """
        old_x, old_y = self.x, self.y

        if action == 0:  # вверх
            self.y -= PLAYER_SPEED
        elif action == 1:  # вниз
            self.y += PLAYER_SPEED
        elif action == 2:  # влево
            self.x -= PLAYER_SPEED
        elif action == 3:  # вправо
            self.x += PLAYER_SPEED
        elif action == 4:  # стоять
            pass

        self._apply_bounds()

        if action in (0, 1, 2, 3) and self.x == old_x and self.y == old_y:
            return True
        return False
