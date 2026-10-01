"""Основной цикл игры."""

import os
import random
import sys

import pygame

from config import (
    BLACK,
    BLUE,
    CONTROL_AI,
    CONTROL_KEYBOARD,
    FPS,
    GAME_WINDOW_POS,
    SHOW_DASHBOARD,
    SHOW_RADAR,
    TARGET_COUNT,
    TARGET_SIZE,
    PLAYER_SIZE,
    WALL_PAIN_ENABLED,
    WHITE,
    WINDOW_HEIGHT,
    WINDOW_WIDTH,
)
from agents import BaseAgent, DummyAgent, NeuralFoodAgent
from dashboard import AIDashboard
from engine import AgentEngine
from experience import RadarReading
from keyboard_player import KeyboardPlayer
from player import Player
from population import Population
from population_view import PopulationView
from radar_view import RadarView
from target import Target


class Game:
    def __init__(self, control_mode=CONTROL_AI):
        self.control_mode = control_mode
        # позиция главного окна до set_mode
        os.environ["SDL_VIDEO_WINDOW_POS"] = (
            f"{GAME_WINDOW_POS[0]},{GAME_WINDOW_POS[1]}"
        )
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("ИИ-агент vs Игра")
        # Классы окон остаются; по умолчанию радар и дашборд не открываются.
        self.radar_view = RadarView() if SHOW_RADAR else None
        self.dashboard = (
            AIDashboard() if SHOW_DASHBOARD and control_mode == CONTROL_AI else None
        )
        self.population_view = None
        self.population = None
        self._overlaps = set()
        self.id_font = pygame.font.Font(None, 28)
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
        avoid = []
        if self.player is not None:
            avoid.append(self.player.rect)
        if self.population is not None:
            avoid.extend(member.player.rect for member in self.population.living())
        x, y = 0, 0
        for _ in range(50):
            x = random.randint(0, WINDOW_WIDTH - TARGET_SIZE)
            y = random.randint(0, WINDOW_HEIGHT - TARGET_SIZE)
            target_rect = pygame.Rect(x, y, TARGET_SIZE, TARGET_SIZE)
            if any(target_rect.colliderect(rect) for rect in avoid):
                continue
            if any(target_rect.colliderect(target.rect) for target in self.targets):
                continue
            self.targets.append(Target(x, y))
            return
        self.targets.append(Target(x, y))

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

    def draw_ui(self, agent: BaseAgent = None):
        score_text = self.font.render(f"Score: {self.player.score}", True, BLACK)
        self.screen.blit(score_text, (10, 10))

        if self.control_mode == CONTROL_KEYBOARD:
            info_text = self.font.render("Arrow keys — move", True, BLUE)
        elif agent is not None:
            mode = getattr(agent, "controller_mode", "ai")
            mode_name = mode.value if hasattr(mode, "value") else str(mode)
            info_text = self.font.render(f"AI mode: {mode_name}", True, BLUE)
        else:
            info_text = self.font.render("AI agent", True, BLUE)
        self.screen.blit(info_text, (10, WINDOW_HEIGHT - 40))

    def draw(self, agent: BaseAgent = None):
        self.screen.fill(WHITE)

        for target in self.targets:
            target.draw(self.screen)

        self.player.draw(self.screen)
        self.draw_ui(agent)

        pygame.display.flip()
        if self.radar_view is not None:
            self.radar_view.draw(self.player.radar)
        if self.dashboard is not None and agent is not None:
            self.dashboard.draw(agent.dashboard_stats())

    def _handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._shutdown()
            elif event.type == pygame.WINDOWCLOSE:
                self._shutdown()

    def _shutdown(self):
        self.running = False
        if self.radar_view is not None:
            self.radar_view.close()
        if self.dashboard is not None:
            self.dashboard.close()
        if self.population_view is not None:
            self.population_view.close()
        pygame.quit()
        sys.exit()

    def _update_radar(self):
        self.player.scan_radar(self.targets)

    def _current_radar_reading(self) -> RadarReading:
        return RadarReading.from_radar(self.player.radar)

    def step(self, action):
        if self.engine is None:
            raise RuntimeError("step() доступен только в режиме CONTROL_AI")

        wall_hit = self.engine.execute(action)
        food_count = self.check_collisions()
        self._update_radar()
        done = False
        state = self.player.get_state(self.targets)
        pain = bool(wall_hit) and WALL_PAIN_ENABLED
        return state, food_count, done, pain

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

            _next_state, food_count, _done, pain = self.step(action)

            agent.observe(
                radar=radar_before,
                action=action,
                food_gained=food_count > 0,
                food_count=food_count,
                pain=pain,
            )
            if isinstance(agent, NeuralFoodAgent):
                agent.note_position(self.player.x, self.player.y)

            if isinstance(agent, NeuralFoodAgent) and agent.needs_training():
                agent.dispatcher.set_training()
                agent._sync_mode()
                self.draw(agent)
                agent.maybe_train()

            self.draw(agent)
            self.clock.tick(FPS)
            pygame.time.delay(30)

    def run_population(self):
        """Несколько особей на одном поле. Радар и дашборд не рисуются."""
        self.reset()
        self.player.x = -1000
        self.player.y = -1000
        self.player.rect.topleft = (-1000, -1000)
        pygame.display.set_caption("Популяция")
        self.population = Population()
        self.population.seed()
        self.population_view = PopulationView()
        self._overlaps = set()

        while self.running:
            self._handle_events()
            train_one = None
            for organism in self.population.living():
                if self._step_organism(organism) and train_one is None:
                    train_one = organism
            self._neighbor_hits()
            for organism in list(self.population.members):
                if organism.alive and organism.vitality.dead:
                    self.population.on_death(organism)
            self.population.tick_birth()
            self._draw_population()
            if train_one is not None:
                train_one.agent.dispatcher.set_training()
                train_one.agent._sync_mode()
                train_one.agent.maybe_train()
            self.clock.tick(FPS)
            pygame.time.delay(30)

    def _step_organism(self, organism) -> bool:
        organism.player.scan_radar(self.targets)
        radar = RadarReading.from_radar(organism.player.radar)
        state = organism.player.get_state(self.targets)
        action = organism.agent.act(radar, state)
        wall_hit = organism.engine.execute(action)
        food_count = self._consume_food(organism.player)
        pain = bool(wall_hit) and WALL_PAIN_ENABLED
        organism.agent.observe(
            radar=radar,
            action=action,
            food_gained=food_count > 0,
            food_count=food_count,
            pain=pain,
        )
        organism.agent.note_position(organism.player.x, organism.player.y)
        organism.vitality.step(food_count, pain, self.population.rng)
        return organism.agent.needs_training()

    def _consume_food(self, player) -> int:
        food_count = 0
        for target in self.targets[:]:
            if player.rect.colliderect(target.rect):
                self.targets.remove(target)
                player.score += 1
                food_count += 1
                self._add_random_target()
        return food_count

    def _neighbor_hits(self):
        living = self.population.living()
        pairs = set()
        for index, left in enumerate(living):
            for right in living[index + 1 :]:
                if not left.player.rect.colliderect(right.player.rect):
                    continue
                key = (left.id, right.id) if left.id < right.id else (right.id, left.id)
                pairs.add(key)
                if key not in self._overlaps:
                    left.vitality.hit_neighbor(self.population.rng)
                    right.vitality.hit_neighbor(self.population.rng)
                    self._separate(left, right)
        self._overlaps = pairs

    def _separate(self, left, right):
        dx = right.player.x - left.player.x
        dy = right.player.y - left.player.y
        if dx == 0 and dy == 0:
            dx = 1
        dist = max(1.0, (dx * dx + dy * dy) ** 0.5)
        push = PLAYER_SIZE / 2
        left.player.x -= dx / dist * push
        left.player.y -= dy / dist * push
        right.player.x += dx / dist * push
        right.player.y += dy / dist * push
        left.player._apply_bounds()
        right.player._apply_bounds()

    def _draw_population(self):
        self.screen.fill(WHITE)
        for target in self.targets:
            target.draw(self.screen)
        for organism in self.population.living():
            organism.player.draw(self.screen)
            label = self.id_font.render(str(organism.id), True, BLACK)
            self.screen.blit(
                label,
                (
                    organism.player.x + 4,
                    organism.player.y + 4,
                ),
            )
        alive = len(self.population.living())
        text = self.font.render(f"Живых: {alive}", True, BLACK)
        self.screen.blit(text, (10, 10))
        pygame.display.flip()
        if self.population_view is not None:
            self.population_view.draw(self.population)

    def run_keyboard(self):
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
