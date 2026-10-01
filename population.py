"""Популяция на одном поле: ранг — время жизни, ребёнок — скрещивание."""

import random
from typing import List, Optional, Sequence, Tuple

import pygame

from agents import NeuralFoodAgent
from config import (
    BIRTH_CERTAIN_AT,
    BIRTH_INTERVAL_MAX,
    BIRTH_INTERVAL_MIN,
    PLAYER_SIZE,
    POPULATION_LIMIT,
    WINDOW_HEIGHT,
    WINDOW_WIDTH,
)
from engine import AgentEngine
from genome import Genome
from player import Player
from vitality import Vitality


def birth_chance(living: int, limit: int = POPULATION_LIMIT, certain_at: int = BIRTH_CERTAIN_AT) -> float:
    """Чем меньше живых на поле, тем выше шанс. При certain_at — всегда, при limit — никогда."""
    if living <= certain_at:
        return 1.0
    if living >= limit:
        return 0.0
    return (limit - living) / float(limit - certain_at)


def next_birth_delay(rng: random.Random) -> int:
    """Случайная длина интервала, 5…20 шагов. Внутри него один случайный момент проверки."""
    span = rng.randint(BIRTH_INTERVAL_MIN, BIRTH_INTERVAL_MAX)
    return rng.randint(BIRTH_INTERVAL_MIN, span)


def choose_weighted(members: Sequence["Organism"], rng: random.Random, exclude: Optional["Organism"] = None) -> "Organism":
    """Случайный родитель. Чем больше ранг (время жизни), тем выше вероятность."""
    pool = [member for member in members if member is not exclude]
    if not pool:
        pool = list(members)
    weights = [max(1, member.rank) for member in pool]
    return rng.choices(pool, weights=weights, k=1)[0]


def drop_worst(members: Sequence["Organism"], limit: int, protect: Optional["Organism"]) -> List["Organism"]:
    """Сверх лимита уходит мёртвая особь с самым коротким временем жизни. Живые остаются."""
    kept = list(members)
    while len(kept) > limit:
        dead = [member for member in kept if not member.alive and member is not protect]
        if not dead:
            break
        worst = min(dead, key=lambda member: (member.rank, -member.id))
        kept.remove(worst)
    return kept


class Organism:
    """Одна особь: геном, тело на поле, сеть и запас прочности."""

    def __init__(
        self,
        organism_id: int,
        genome: Genome,
        player: Player,
        agent: NeuralFoodAgent,
        parents: Tuple[int, ...],
        origin: str,
    ):
        self.id = organism_id
        self.genome = genome
        self.player = player
        self.agent = agent
        self.engine = AgentEngine(player)
        self.vitality = Vitality()
        self.alive = True
        self.lifespan = 0
        self.parents = parents
        self.origin = origin

    @property
    def rank(self) -> int:
        if self.alive:
            return self.vitality.age
        return self.lifespan

    def mark_dead(self) -> None:
        self.alive = False
        self.lifespan = self.vitality.age


class Population:
    """
    Конечный список особей.

    Первые две — случайные геномы. Третья — их скрещивание без мутации.
    Дальше ребёнок появляется в случайный шаг интервала 5…20.
    Чем меньше живых на поле, тем выше шанс: при двух — всегда, при восьми — никогда.
    Родители выбираются с вероятностью по рангу.
    Из списка выбывают только мёртвые. Если место занято живыми, новая особь не рождается.
    """

    def __init__(self, rng: Optional[random.Random] = None, limit: int = POPULATION_LIMIT):
        if limit < 3:
            raise ValueError("в популяции нужно место минимум для двух основателей и их ребёнка")
        self.rng = rng if rng is not None else random.Random()
        self.limit = limit
        self.members: List[Organism] = []
        self._next_id = 1
        self.newest: Optional[Organism] = None
        self.births = 0
        self._steps_until_birth = 0

    def living(self) -> List[Organism]:
        return [member for member in self.members if member.alive]

    def seed(self) -> None:
        first = self._add(self._birth_random())
        second = self._add(self._birth_random())
        self._add(self._birth_crossover(first, second, mutate=False))
        self._schedule_birth()

    def on_death(self, organism: Organism) -> None:
        organism.mark_dead()

    def tick_birth(self) -> bool:
        """Один шаг мира. В случайный момент интервала — жребий новой особи."""
        if self._steps_until_birth > 0:
            self._steps_until_birth -= 1
        if self._steps_until_birth > 0:
            return False
        born = self._try_birth()
        self._schedule_birth()
        return born

    def _schedule_birth(self) -> None:
        self._steps_until_birth = next_birth_delay(self.rng)

    def _has_room(self) -> bool:
        if len(self.members) < self.limit:
            return True
        return any(not member.alive for member in self.members)

    def _try_birth(self) -> bool:
        if len(self.members) < 2:
            return False
        # Восемь живых — шанс 0. Полный список без мёртвых тоже не пускает новорождённого.
        if len(self.living()) >= self.limit or not self._has_room():
            return False
        if self.rng.random() >= birth_chance(len(self.living()), self.limit):
            return False
        self._add(self._birth_from_selection(mutate=True))
        self.members = drop_worst(self.members, self.limit, self.newest)
        return True

    def _add(self, organism: Organism) -> Organism:
        self.members.append(organism)
        self.newest = organism
        self.births += 1
        return organism

    def _birth_random(self) -> Organism:
        return self._spawn(Genome.randomized(self.rng), (), "random")

    def _birth_from_selection(self, mutate: bool) -> Organism:
        first = choose_weighted(self.members, self.rng)
        second = choose_weighted(self.members, self.rng, exclude=first)
        return self._birth_crossover(first, second, mutate=mutate)

    def _birth_crossover(self, first: Organism, second: Organism, mutate: bool) -> Organism:
        genome = first.genome.crossover(second.genome, self.rng)
        if mutate:
            genome.mutate(self.rng, n_genes=1)
        return self._spawn(genome, (first.id, second.id), "cross")

    def _spawn(self, genome: Genome, parents: Tuple[int, ...], origin: str) -> Organism:
        organism_id = self._next_id
        self._next_id += 1
        x, y = self._random_pos()
        player = Player(x, y)
        agent = NeuralFoodAgent(genome=genome, birth_rng=self.rng)
        return Organism(organism_id, genome, player, agent, parents, origin)

    def _random_pos(self) -> Tuple[int, int]:
        occupied = [member.player.rect for member in self.living()]
        x, y = 0, 0
        for _ in range(40):
            x = self.rng.randint(0, WINDOW_WIDTH - PLAYER_SIZE)
            y = self.rng.randint(0, WINDOW_HEIGHT - PLAYER_SIZE)
            rect = pygame.Rect(x, y, PLAYER_SIZE, PLAYER_SIZE)
            if any(rect.colliderect(other) for other in occupied):
                continue
            return x, y
        return x, y
