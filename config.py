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

# Нейросеть и режимы управления
NN_HIDDEN_1 = 128
NN_HIDDEN_2 = 64
NN_EPOCHS = 8
NN_BATCH_SIZE = 32
NN_LEARNING_RATE = 1e-3

# Оценщик: окно последних нейросетевых попыток
EVAL_WINDOW = 12
EVAL_MIN_SAMPLES = 5
EVAL_MIN_SUCCESS_RATE = 0.25  # ниже → случайное блуждание

# Обучение: минимум успешных шагов в буфере и доля очистки после train
TRAIN_MIN_SUCCESS_STEPS = 40
EXPERIENCE_KEEP_FRACTION = 0.4  # после обучения оставляем последние 40%
# Дообучение в режиме neural: после стольких успешных «нашёл еду»
TRAIN_EVERY_N_NEURAL_FOODS = 3

# Дашборд
DASHBOARD_WIDTH = 420
DASHBOARD_HEIGHT = 360
