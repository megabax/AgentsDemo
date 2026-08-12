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
from agents import BaseAgent, DummyAgent
from engine import AgentEngine
from experience import RadarReading
from keyboard_player import KeyboardPlayer
from player import Player
from radar_view import RadarView
from target import Target


class Game:
    def __init__(self, control_mode=CONTROL_AI):
        self.control_mode = control_mode
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("ИИ-агент vs Игра (шаблон)")
        self.radar_view = RadarView()
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        self.engine = None
        self.reset()

    def reset(self):
        player_x = WINDOW_WIDTH // 2 - PLAYER_SIZE // 2
        player_y = WINDOW_HEIGHT // 2 - PLAYER_SIZE // 2
        if self.control_mode == CONTROL_KEYBOARD:
            self.player = KeyboardPlayer(player_x, player_y)
            self.engine = None
        else:
            self.player = Player(player_x, player_y)
            self.engine = AgentEngine(self.player)

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
        """Проверка еды. Возвращает число съеденных целей на этом шаге."""
        food_count = 0
        for target in self.targets[:]:
            if self.player.rect.colliderect(target.rect):
                self.targets.remove(target)
                self.player.score += 1
                food_count += 1
                self._add_random_target()
        return food_count

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
        self.radar_view.draw(self.player.radar)

    def _handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._shutdown()
            elif event.type == pygame.WINDOWCLOSE:
                self._shutdown()

    def _shutdown(self):
        self.running = False
        self.radar_view.close()
        pygame.quit()
        sys.exit()

    def _update_radar(self):
        self.player.scan_radar(self.targets)

    def _current_radar_reading(self) -> RadarReading:
        return RadarReading.from_radar(self.player.radar)

    def step(self, action):
        """
        Один шаг среды: движок двигает агента, затем еда и радар.
        Возвращает (state, food_count, done).
        """
        if self.engine is None:
            raise RuntimeError("step() доступен только в режиме CONTROL_AI")

        self.engine.execute(action)
        food_count = self.check_collisions()
        self._update_radar()
        done = False
        state = self.player.get_state(self.targets)
        return state, food_count, done

    def run_with_ai(self, agent: BaseAgent = None):
        if self.control_mode != CONTROL_AI:
            raise ValueError("run_with_ai только при control_mode=CONTROL_AI (класс Player).")

        if agent is None:
            agent = DummyAgent()

        self.reset()
        agent.reset()
        self.engine.bind(self.player)
        self._update_radar()

        while self.running:
            self._handle_events()

            radar_before = self._current_radar_reading()
            state = self.player.get_state(self.targets)
            action = agent.act(radar_before, state)

            _next_state, food_count, _done = self.step(action)

            agent.observe(
                radar=radar_before,
                action=action,
                food_gained=food_count > 0,
                food_count=food_count,
            )
            # обучение-заглушка; реальная логика появится позже
            agent.learn()

            self.draw()
            self.clock.tick(FPS)
            pygame.time.delay(30)

    def run_keyboard(self):
        """Игра с управлением стрелками (класс KeyboardPlayer)."""
        if self.control_mode != CONTROL_KEYBOARD:
            raise ValueError("run_keyboard только при control_mode=CONTROL_KEYBOARD.")

        self.reset()
        self._update_radar()

        while self.running:
            self._handle_events()
            self.player.update_from_keys()
            self.check_collisions()
            self._update_radar()
            self.draw()
            self.clock.tick(FPS)
