"""Основной цикл игры."""

import random
import sys

import pygame

from config import (
    BLACK,
    BLUE,
    CONTROL_AI,
    CONTROL_KEYBOARD,
    FPS,
    TARGET_COUNT,
    TARGET_SIZE,
    PLAYER_SIZE,
    WHITE,
    WINDOW_HEIGHT,
    WINDOW_WIDTH,
)
from dummy_agent import DummyAgent
from keyboard_player import KeyboardPlayer
from player import Player
from target import Target


class Game:
    def __init__(self, control_mode=CONTROL_AI):
        self.control_mode = control_mode
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("ИИ-агент vs Игра (шаблон)")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        self.reset()

    def reset(self):
        player_x = WINDOW_WIDTH // 2 - PLAYER_SIZE // 2
        player_y = WINDOW_HEIGHT // 2 - PLAYER_SIZE // 2
        if self.control_mode == CONTROL_KEYBOARD:
            self.player = KeyboardPlayer(player_x, player_y)
        else:
            self.player = Player(player_x, player_y)

        self.targets = []
        for _ in range(TARGET_COUNT):
            self._add_random_target()

        self.running = True
        self.frame_count = 0

    def _add_random_target(self):
        while True:
            x = random.randint(0, WINDOW_WIDTH - TARGET_SIZE)
            y = random.randint(0, WINDOW_HEIGHT - TARGET_SIZE)
            target_rect = pygame.Rect(x, y, TARGET_SIZE, TARGET_SIZE)
            if not target_rect.colliderect(self.player.rect):
                self.targets.append(Target(x, y))
                break

    def check_collisions(self):
        for target in self.targets[:]:
            if self.player.rect.colliderect(target.rect):
                self.targets.remove(target)
                self.player.score += 1
                self._add_random_target()

    def draw_ui(self):
        score_text = self.font.render(f"Score: {self.player.score}", True, BLACK)
        self.screen.blit(score_text, (10, 10))

        if self.control_mode == CONTROL_KEYBOARD:
            info_text = self.font.render("Arrow keys — move", True, BLUE)
        else:
            info_text = self.font.render("AI Agent Placeholder (random movement)", True, BLUE)
        self.screen.blit(info_text, (10, WINDOW_HEIGHT - 40))

    def draw(self):
        self.screen.fill(WHITE)

        for target in self.targets:
            target.draw(self.screen)

        self.player.draw(self.screen)
        self.draw_ui()

        pygame.display.flip()

    def step(self, action):
        old_score = self.player.score

        self.player.update(action)

        self.check_collisions()

        reward = self.player.score - old_score

        done = False

        state = self.player.get_state(self.targets)

        return state, reward, done

    def run_with_ai(self, agent=None):
        if self.control_mode != CONTROL_AI:
            raise ValueError("run_with_ai только при control_mode=CONTROL_AI (класс Player).")

        if agent is None:
            agent = DummyAgent()

        self.reset()

        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    pygame.quit()
                    sys.exit()

            state = self.player.get_state(self.targets)

            action = agent.act(state)

            next_state, reward, done = self.step(action)

            agent.learn(state, action, reward, next_state, done)

            self.draw()
            self.clock.tick(FPS)

            pygame.time.delay(30)

    def run_keyboard(self):
        """Игра с управлением стрелками (класс KeyboardPlayer)."""
        if self.control_mode != CONTROL_KEYBOARD:
            raise ValueError("run_keyboard только при control_mode=CONTROL_KEYBOARD.")

        self.reset()

        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    pygame.quit()
                    sys.exit()

            self.player.update_from_keys()
            self.check_collisions()

            self.draw()
            self.clock.tick(FPS)
