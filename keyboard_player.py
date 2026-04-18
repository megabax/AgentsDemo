"""Игрок с управлением стрелками (курсор)."""

import pygame

from config import (
    PLAYER_SIZE,
    PLAYER_SPEED,
    RED,
    WINDOW_HEIGHT,
    WINDOW_WIDTH,
)


class KeyboardPlayer:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.size = PLAYER_SIZE
        self.rect = pygame.Rect(x, y, self.size, self.size)
        self.score = 0

    def update_from_keys(self):
        """Движение по зажатым стрелкам (можно по диагонали)."""
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            self.y -= PLAYER_SPEED
        if keys[pygame.K_DOWN]:
            self.y += PLAYER_SPEED
        if keys[pygame.K_LEFT]:
            self.x -= PLAYER_SPEED
        if keys[pygame.K_RIGHT]:
            self.x += PLAYER_SPEED

        self.x = max(0, min(WINDOW_WIDTH - self.size, self.x))
        self.y = max(0, min(WINDOW_HEIGHT - self.size, self.y))

        self.rect.topleft = (self.x, self.y)

    def draw(self, screen):
        pygame.draw.rect(screen, RED, self.rect)

    def get_state(self, targets):
        player_pos = (self.x, self.y)
        targets_pos = [(t.x, t.y) for t in targets]
        return {
            "player_pos": player_pos,
            "targets_pos": targets_pos,
            "score": self.score,
        }
