"""Цель, которую нужно собирать."""

import pygame

from config import GREEN, TARGET_SIZE


class Target:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.size = TARGET_SIZE
        self.rect = pygame.Rect(x, y, self.size, self.size)

    def draw(self, screen):
        pygame.draw.rect(screen, GREEN, self.rect)
