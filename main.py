"""
Простой шаблон проекта для ИИ-агента в компьютерной игре.
Игра: управление красным квадратом, который должен собирать зелёные цели.
Агент-заглушка: просто двигается случайным образом.
"""

import pygame
import random
import sys

# Инициализация Pygame
pygame.init()

# Константы
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
FPS = 60

# Цвета
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
BLACK = (0, 0, 0)

# Параметры игрока (агента)
PLAYER_SIZE = 40
PLAYER_SPEED = 5

# Параметры цели
TARGET_SIZE = 30
TARGET_COUNT = 5


class Player:
    """Игрок (управляемый ИИ-агентом)"""
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.size = PLAYER_SIZE
        self.rect = pygame.Rect(x, y, self.size, self.size)
        self.score = 0

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

        # Границы окна
        self.x = max(0, min(WINDOW_WIDTH - self.size, self.x))
        self.y = max(0, min(WINDOW_HEIGHT - self.size, self.y))

        self.rect.topleft = (self.x, self.y)

    def draw(self, screen):
        pygame.draw.rect(screen, RED, self.rect)

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
            "score": self.score
        }


class Target:
    """Цель, которую нужно собирать"""
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.size = TARGET_SIZE
        self.rect = pygame.Rect(x, y, self.size, self.size)

    def draw(self, screen):
        pygame.draw.rect(screen, GREEN, self.rect)


class Game:
    """Основной класс игры"""
    def __init__(self):
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("ИИ-агент vs Игра (шаблон)")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        self.reset()

    def reset(self):
        """Сброс игры в начальное состояние"""
        # Создание игрока в центре
        player_x = WINDOW_WIDTH // 2 - PLAYER_SIZE // 2
        player_y = WINDOW_HEIGHT // 2 - PLAYER_SIZE // 2
        self.player = Player(player_x, player_y)

        # Создание целей в случайных местах
        self.targets = []
        for _ in range(TARGET_COUNT):
            self._add_random_target()

        self.running = True
        self.frame_count = 0

    def _add_random_target(self):
        """Добавление случайной цели (не на игрока)"""
        while True:
            x = random.randint(0, WINDOW_WIDTH - TARGET_SIZE)
            y = random.randint(0, WINDOW_HEIGHT - TARGET_SIZE)
            target_rect = pygame.Rect(x, y, TARGET_SIZE, TARGET_SIZE)
            if not target_rect.colliderect(self.player.rect):
                self.targets.append(Target(x, y))
                break

    def check_collisions(self):
        """Проверка столкновений игрока с целями"""
        for target in self.targets[:]:
            if self.player.rect.colliderect(target.rect):
                self.targets.remove(target)
                self.player.score += 1
                self._add_random_target()

    def draw_ui(self):
        """Отрисовка интерфейса"""
        score_text = self.font.render(f"Score: {self.player.score}", True, BLACK)
        self.screen.blit(score_text, (10, 10))

        # Информация для разработчика
        info_text = self.font.render("AI Agent Placeholder (random movement)", True, BLUE)
        self.screen.blit(info_text, (10, WINDOW_HEIGHT - 40))

    def draw(self):
        """Отрисовка всего на экране"""
        self.screen.fill(WHITE)

        for target in self.targets:
            target.draw(self.screen)

        self.player.draw(self.screen)
        self.draw_ui()

        pygame.display.flip()

    def step(self, action):
        """
        Выполнение одного шага игры.
        action: действие от ИИ-агента
        Возвращает: состояние, награда, done
        """
        # Сохраняем старый счёт для вычисления награды
        old_score = self.player.score

        # Обновляем игрока действием
        self.player.update(action)

        # Проверяем столкновения
        self.check_collisions()

        # Вычисляем награду
        reward = self.player.score - old_score

        # Игра не заканчивается (можно добавить условие)
        done = False

        # Получаем новое состояние
        state = self.player.get_state(self.targets)

        return state, reward, done

    def run_with_ai(self, agent=None):
        """
        Запуск игры с ИИ-агентом.
        Если agent = None, используется агент-заглушка.
        """
        if agent is None:
            agent = DummyAgent()

        self.reset()

        while self.running:
            # Обработка событий Pygame (выход)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    pygame.quit()
                    sys.exit()

            # Получаем текущее состояние
            state = self.player.get_state(self.targets)

            # Агент выбирает действие
            action = agent.act(state)

            # Выполняем шаг игры
            next_state, reward, done = self.step(action)

            # Обучаем агента (если нужно)
            agent.learn(state, action, reward, next_state, done)

            # Отрисовка
            self.draw()
            self.clock.tick(FPS)

            # Небольшая задержка для наглядности
            pygame.time.delay(30)


class DummyAgent:
    """
    Агент-заглушка.
    Вместо настоящего ИИ просто выбирает случайное действие.
    Это место, куда вы будете встраивать своего ИИ-агента.
    """

    def __init__(self):
        # Здесь можно инициализировать модель, память и т.д.
        self.actions = [0, 1, 2, 3, 4]  # вверх, вниз, влево, вправо, стоять

    def act(self, state):
        """
        Выбор действия на основе текущего состояния.
        state: словарь с информацией о игре
        """
        # ЗАГЛУШКА: случайное действие
        # Здесь будет ваш ИИ (нейросеть, Q-learning, и т.д.)
        return random.choice(self.actions)

    def learn(self, state, action, reward, next_state, done):
        """
        Обучение агента на основе опыта.
        Для заглушки ничего не делаем.
        """
        # Здесь будет обучение вашего агента
        pass


def main():
    """Главная функция"""
    game = Game()

    # Создаём агента-заглушку (вместо настоящего ИИ)
    dummy_ai = DummyAgent()

    # Запускаем игру с агентом
    game.run_with_ai(dummy_ai)


if __name__ == "__main__":
    main()