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


class BasePlayer:
    """Общее состояние, отрисовка, границы и снимок state для агента."""

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.size = PLAYER_SIZE
        self.rect = pygame.Rect(x, y, self.size, self.size)
        self.score = 0

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

    def draw(self, screen):
        pygame.draw.rect(screen, RED, self.rect)

    def get_state(self, targets):
        """
        Получение текущего состояния для ИИ-агента.
        Возвращает: позицию игрока, позиции целей, количество очков
        """
        return {
            "player_pos": (self.x, self.y),
            "targets_pos": [(t.x, t.y) for t in targets],
            "score": self.score,
        }


class Player(BasePlayer):
    def update(self, action):
        """
        Обновление позиции игрока на основе действия агента.
        action: 0 - вверх, 1 - вниз, 2 - влево, 3 - вправо, 4 - стоять
        """
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
