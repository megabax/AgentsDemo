"""Круговой радар: лучи измеряют расстояние и цвет препятствий."""

import math

from config import (
    GREEN,
    RADAR_MAX_RANGE,
    RADAR_RAY_COUNT,
    RADAR_STEP,
    WHITE,
    WINDOW_HEIGHT,
    WINDOW_WIDTH,
)


class Radar:
    """
    Обходит полный круг лучами из центра игрока.
    На каждый луч: расстояние до первого попадания и цвет объекта.
    Target -> зелёный, граница окна (стена) -> белый, пустота -> чёрный.

    Интенсивность цвета падает как (1 − d/R)², где R — диагональ поля
    (дальность зрения). На R сигнал обнуляется; ближе — ярче.
    """

    def __init__(
        self,
        ray_count=RADAR_RAY_COUNT,
        max_range=RADAR_MAX_RANGE,
        step=RADAR_STEP,
    ):
        self.ray_count = ray_count
        self.max_range = max_range
        self.step = step
        self.distances = [max_range] * ray_count
        self.colors = [(0, 0, 0)] * ray_count

    def _intensity(self, distance):
        """
        Квадратичное затухание (1 − d/R)².

        Чистый закон 1/r² слишком резко гасит сигнал на пиксельной шкале:
        уже на десятках пикселей цвет был бы почти чёрным.
        (1 − d/R)² быстрее линейного (как энергия), и на дальности
        зрения R (диагональ поля) интенсивность точно равна нулю.
        """
        R = float(self.max_range)
        if R <= 0:
            return 0.0
        t = min(1.0, max(0.0, float(distance) / R))
        return (1.0 - t) ** 2

    def _attenuate(self, color, distance):
        intensity = self._intensity(distance)
        return tuple(int(channel * intensity) for channel in color)

    def scan(self, origin_x, origin_y, targets):
        """Полный обход 360°; origin — центр игрока."""
        for i in range(self.ray_count):
            angle = 2 * math.pi * i / self.ray_count
            dx = math.cos(angle)
            dy = math.sin(angle)

            hit_dist = self.max_range
            hit_color = (0, 0, 0)

            for dist in range(self.step, self.max_range + 1, self.step):
                x = origin_x + dx * dist
                y = origin_y + dy * dist

                if x < 0 or x >= WINDOW_WIDTH or y < 0 or y >= WINDOW_HEIGHT:
                    hit_dist = dist
                    hit_color = WHITE
                    break

                hit_target = False
                for target in targets:
                    if target.rect.collidepoint(x, y):
                        hit_dist = dist
                        hit_color = GREEN
                        hit_target = True
                        break
                if hit_target:
                    break

            self.distances[i] = hit_dist
            self.colors[i] = self._attenuate(hit_color, hit_dist)

    def as_dict(self):
        """Снимок показаний для будущего state агента."""
        return {
            "distances": list(self.distances),
            "colors": list(self.colors),
        }
