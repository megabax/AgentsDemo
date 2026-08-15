"""Дашборд режима ИИ: neural / random / training и статистика опыта."""

import pygame
from pygame._sdl2.video import Renderer, Texture, Window

from config import DASHBOARD_HEIGHT, DASHBOARD_WIDTH


class AIDashboard:
    def __init__(self):
        self.window = Window("AI Dashboard", size=(DASHBOARD_WIDTH, DASHBOARD_HEIGHT))
        self.renderer = Renderer(self.window)
        self.surface = pygame.Surface((DASHBOARD_WIDTH, DASHBOARD_HEIGHT))
        self.font = pygame.font.Font(None, 28)
        self.small = pygame.font.Font(None, 24)

    def draw(self, stats: dict) -> None:
        self.surface.fill((28, 30, 36))
        mode = stats.get("mode", "?")
        mode_colors = {
            "neural": (80, 180, 120),
            "random": (220, 160, 60),
            "training": (90, 140, 230),
        }
        accent = mode_colors.get(mode, (180, 180, 180))

        pygame.draw.rect(self.surface, accent, pygame.Rect(0, 0, DASHBOARD_WIDTH, 48))
        title = self.font.render(f"Mode: {mode.upper()}", True, (20, 20, 24))
        self.surface.blit(title, (16, 12))

        lines = [
            f"Experience steps: {stats.get('experience_steps', 0)}",
            f"Closed attempts: {stats.get('attempts', 0)}",
            f"  successes: {stats.get('successes', 0)}  failures: {stats.get('failures', 0)}",
            f"Trainings: {stats.get('train_count', 0)}",
            f"Last train samples: {stats.get('last_train_samples', '-')}",
            f"Last loss / acc: {stats.get('last_loss', '-')} / {stats.get('last_accuracy', '-')}",
            f"Buffer cleanups: {stats.get('cleanup_count', 0)}",
            f"Last cleanup removed: {stats.get('last_cleanup_removed', 0)}",
            f"NN eval rate: {stats.get('eval_success_rate', '-')}",
            f"Eval samples: {stats.get('eval_samples', 0)} (need ≥ {stats.get('eval_min_samples', 0)})",
            f"Switch→random?: {stats.get('switch_to_random', False)}",
            f"Food found (total): {stats.get('food_total', 0)}",
            f"Random foods since NN fail: {stats.get('random_foods', 0)}",
            f"Neural foods since train: {stats.get('neural_foods_since_train', 0)}"
            f" / {stats.get('train_every_n_neural_foods', '?')}",
        ]

        y = 64
        for line in lines:
            text = self.small.render(str(line), True, (220, 220, 225))
            self.surface.blit(text, (16, y))
            y += 22

        hint = self.small.render(
            "neural=policy  random=explore  training=fit",
            True,
            (140, 145, 155),
        )
        self.surface.blit(hint, (16, DASHBOARD_HEIGHT - 32))

        texture = Texture.from_surface(self.renderer, self.surface)
        self.renderer.clear()
        texture.draw()
        self.renderer.present()

    def close(self) -> None:
        self.window.destroy()
