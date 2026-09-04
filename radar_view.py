"""Второе окно: три графика RGB по показаниям радара."""

import pygame
from pygame._sdl2.video import Renderer, Texture, Window

from config import (
    BLUE,
    GREEN,
    RADAR_VIEW_HEIGHT,
    RADAR_VIEW_WIDTH,
    RADAR_WINDOW_POS,
    RED,
)


class RadarView:
    """Окно с тремя графиками каналов R, G, B вдоль угла обхода."""

    def __init__(self):
        self.window = Window(
            "Radar RGB",
            size=(RADAR_VIEW_WIDTH, RADAR_VIEW_HEIGHT),
        )
        self.window.position = RADAR_WINDOW_POS
        self.renderer = Renderer(self.window)
        self.surface = pygame.Surface((RADAR_VIEW_WIDTH, RADAR_VIEW_HEIGHT))
        self.font = pygame.font.Font(None, 28)
        self._margin = 40
        self._label_w = 28

    def draw(self, radar):
        self.surface.fill((30, 30, 34))

        channels = (
            ("R", RED, 0),
            ("G", GREEN, 1),
            ("B", BLUE, 2),
        )
        plot_h = (RADAR_VIEW_HEIGHT - self._margin * 2) // 3

        for idx, (label, color, channel) in enumerate(channels):
            top = self._margin + idx * plot_h
            self._draw_channel(
                label,
                color,
                channel,
                radar,
                top,
                plot_h - 12,
            )

        texture = Texture.from_surface(self.renderer, self.surface)
        self.renderer.clear()
        texture.draw()
        self.renderer.present()

    def _draw_channel(self, label, color, channel, radar, top, height):
        left = self._margin + self._label_w
        width = RADAR_VIEW_WIDTH - left - self._margin
        rect = pygame.Rect(left, top, width, height)

        pygame.draw.rect(self.surface, (45, 45, 50), rect)
        pygame.draw.rect(self.surface, (80, 80, 90), rect, 1)

        label_surf = self.font.render(label, True, color)
        self.surface.blit(
            label_surf,
            (self._margin - 4, top + height // 2 - label_surf.get_height() // 2),
        )

        n = max(1, radar.ray_count - 1)
        points = []
        for i, rgb in enumerate(radar.colors):
            x = left + int(i / n * (width - 1))
            y = top + height - 1 - int(rgb[channel] / 255 * (height - 1))
            points.append((x, y))

        if len(points) >= 2:
            pygame.draw.lines(self.surface, color, False, points, 2)

        tip = self.font.render("0", True, (140, 140, 150))
        self.surface.blit(tip, (left - 18, top + height - 9))
        tip255 = self.font.render("255", True, (140, 140, 150))
        self.surface.blit(tip255, (left - 28, top - 2))

    def close(self):
        self.window.destroy()
