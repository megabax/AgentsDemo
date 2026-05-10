"""Игрок (управляемый ИИ-агентом)."""

import pygame

from config import (
    PLAYER_SIZE,
    PLAYER_SPEED,
    PLAYER_SPRITE_PATH,
    WHITE,
    WINDOW_HEIGHT,
    WINDOW_WIDTH,
)


class Player:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.size = PLAYER_SIZE
        self.rect = pygame.Rect(x, y, self.size, self.size)
        self.score = 0

        img = pygame.image.load(PLAYER_SPRITE_PATH).convert()
        img.set_colorkey(WHITE)
        self.image = pygame.transform.smoothscale(img, (self.size, self.size))

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

        # Новая логика Б:
        if self.rect.right > WINDOW_WIDTH:
            self.rect.x = 0  # телепортация налево
        if self.rect.left < 0:
            self.rect.x = WINDOW_WIDTH - PLAYER_SIZE  # телепортация направо

        self.x = max(0, min(WINDOW_WIDTH - self.size, self.x))
        self.y = max(0, min(WINDOW_HEIGHT - self.size, self.y))

        self.rect.topleft = (self.x, self.y)

    def draw(self, screen):
        screen.blit(self.image, self.rect)

    def get_state(self, targets):
        """
        Получение текущего состояния для ИИ-агента.
        Возвращает: позицию игрока, позиции целей, количество очков
        """
        player_pos = (self.x, self.y)
        targets_pos = [(t.x, t.y) for t in targets]
        return {
            "player_pos": player_pos,
            "targets_pos": targets_pos,
            "score": self.score,
        }
