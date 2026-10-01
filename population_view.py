"""Окно списка особей: время жизни и основные гены."""

import pygame
from pygame._sdl2.video import Renderer, Texture, Window

from config import POPULATION_VIEW_HEIGHT, POPULATION_VIEW_WIDTH, POPULATION_WINDOW_POS
from population import Population


CAUSES = {
    "hunger": "голод",
    "wall": "стена",
    "neighbor": "сосед",
    "age": "старость",
}


class PopulationView:
    def __init__(self):
        self.window = Window(
            "Популяция",
            size=(POPULATION_VIEW_WIDTH, POPULATION_VIEW_HEIGHT),
        )
        self.window.position = POPULATION_WINDOW_POS
        self.renderer = Renderer(self.window)
        self.surface = pygame.Surface((POPULATION_VIEW_WIDTH, POPULATION_VIEW_HEIGHT))
        self.font = pygame.font.Font(None, 24)
        self.small = pygame.font.Font(None, 20)

    def draw(self, population: Population) -> None:
        self.surface.fill((28, 30, 36))
        title = self.font.render(
            f"Популяция  {len(population.members)}/{population.limit}"
            f"   рождений {population.births}",
            True,
            (230, 230, 235),
        )
        self.surface.blit(title, (16, 12))
        hint = self.small.render(
            "Ранг — время жизни. Чем оно больше, тем выше шанс стать родителем.",
            True,
            (160, 165, 175),
        )
        self.surface.blit(hint, (16, 40))
        birth = self.small.render(
            "Новая особь — случайный шаг интервала 5–20. При 2 живых всегда, при 8 никогда.",
            True,
            (160, 165, 175),
        )
        self.surface.blit(birth, (16, 58))

        ranked = sorted(population.members, key=lambda member: (-member.rank, member.id))
        top = 84
        row_h = 62
        for index, member in enumerate(ranked):
            self._draw_row(member, top + index * row_h)

        texture = Texture.from_surface(self.renderer, self.surface)
        self.renderer.clear()
        texture.draw()
        self.renderer.present()

    def _draw_row(self, member, top: int) -> None:
        alive = member.alive
        accent = (80, 180, 120) if alive else (120, 120, 130)
        pygame.draw.rect(self.surface, accent, pygame.Rect(12, top, 6, 52))
        status = "жива" if alive else "мертва"
        cause = CAUSES.get(member.vitality.cause, "")
        tail = f"  {cause}" if (not alive and cause) else ""
        parents = "—"
        if member.parents:
            parents = "×".join(f"#{parent_id}" for parent_id in member.parents)
        line = (
            f"#{member.id}  {status}  жизнь {member.rank}"
            f"  hp {int(member.vitality.hp)}"
            f"  еда {member.agent.food_total}"
            f"  родители {parents}{tail}"
        )
        self.surface.blit(self.font.render(line, True, (230, 230, 235)), (26, top + 4))

        genome = member.genome
        behavior = genome.behavior
        birth = "копия" if member.agent.birth_source == "inherit" else "случ"
        details = (
            f"сеть {genome.architecture.hidden_1}/{genome.architecture.hidden_2}"
            f"  история {genome.architecture.history_len}"
            f"  липкость {behavior.gene('neural_sticky_steps').value}"
            f"  серия {behavior.gene('random_walk_min_steps').value}"
            f"–{behavior.gene('random_walk_max_steps').value}"
            f"  inherit {genome.innate_weights.inherit_weights:.2f}/{birth}"
        )
        self.surface.blit(self.small.render(details, True, (170, 176, 186)), (26, top + 30))

    def close(self) -> None:
        self.window.destroy()
