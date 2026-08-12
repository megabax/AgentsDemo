"""Общие константы игры."""

WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
FPS = 60

WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
BLACK = (0, 0, 0)

TARGET_SIZE = 30
PLAYER_SIZE = TARGET_SIZE
PLAYER_SPEED = 5
TARGET_COUNT = 5

# Режим управления: CONTROL_AI — класс Player и ИИ-агент; CONTROL_KEYBOARD — KeyboardPlayer и стрелки
CONTROL_AI = "ai"
CONTROL_KEYBOARD = "keyboard"

# Режимы проверки границ
BOUNDARY_MODE_BOUNCE = 0      # Останов у края (без скорости направления)
BOUNDARY_MODE_WRAP = 1        # Телепортация (появляется с противоположной стороны)


# Выберите режим (по умолчанию - отскок)
BOUNDARY_MODE = BOUNDARY_MODE_BOUNCE

# Радар (зрение игрока)
RADAR_RAY_COUNT = 180
# Дальность зрения = диагональ поля; на ней интенсивность → 0
RADAR_MAX_RANGE = int((WINDOW_WIDTH**2 + WINDOW_HEIGHT**2) ** 0.5)
RADAR_STEP = 2
RADAR_VIEW_WIDTH = 640
RADAR_VIEW_HEIGHT = 480

# Попытка найти еду: лимит шагов до исхода «не дошёл»
ATTEMPT_MAX_STEPS = 200
