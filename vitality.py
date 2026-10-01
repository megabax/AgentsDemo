"""Запас прочности: голод, стена, сосед и старость. Еда лечит."""

import random

from config import (
    AGE_DAMAGE,
    AGE_GROWTH,
    FOOD_HEAL,
    HUNGER_DAMAGE,
    HUNGER_GROWTH,
    NEIGHBOR_DAMAGE,
    VITALITY_MAX,
    WALL_DAMAGE,
)


class Vitality:
    """
    Урон всегда случайный: константа ситуации × random(0..1) × коэффициент.
    У голода коэффициент растёт с длительностью голода, у старости — с возрастом.
    Смерть наступает, когда запас падает до нуля. Причина — последний удар.
    """

    def __init__(self, max_hp: float = VITALITY_MAX):
        self.max_hp = float(max_hp)
        self.hp = float(max_hp)
        self.age = 0
        self.hunger_steps = 0
        self.dead = False
        self.cause = ""

    def step(self, food_count: int, wall: bool, rng: random.Random) -> None:
        if self.dead:
            return
        self.age += 1
        if food_count > 0:
            self.hunger_steps = 0
            self.heal(FOOD_HEAL * food_count)
        else:
            self.hunger_steps += 1
            growth = 1.0 + HUNGER_GROWTH * self.hunger_steps
            self.apply("hunger", HUNGER_DAMAGE * rng.random() * growth)
        if wall:
            self.apply("wall", WALL_DAMAGE * rng.random())
        age_growth = 1.0 + AGE_GROWTH * self.age
        self.apply("age", AGE_DAMAGE * rng.random() * age_growth)

    def hit_neighbor(self, rng: random.Random) -> None:
        self.apply("neighbor", NEIGHBOR_DAMAGE * rng.random())

    def heal(self, amount: float) -> None:
        if self.dead:
            return
        self.hp = min(self.max_hp, self.hp + amount)

    def apply(self, cause: str, amount: float) -> None:
        if self.dead:
            return
        self.hp -= float(amount)
        if self.hp <= 0:
            self.hp = 0.0
            self.dead = True
            self.cause = cause
