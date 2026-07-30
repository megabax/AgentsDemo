"""Игрок с управлением стрелками (курсор)."""

import pygame

from config import PLAYER_SPEED
from player import BasePlayer


class KeyboardPlayer(BasePlayer):
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

        self._apply_bounds()
