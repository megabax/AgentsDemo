"""Допустимые значения генов. Их же описывает genome/ENCODING.md."""

GENOME_VERSION = 1

# Сетка архитектур первых поколений. Вне этих значений ген не хранится.
HIDDEN_WIDTHS = (64, 128, 256)
HISTORY_LENGTHS = (1, 3, 5)

# Вход одного кадра: 4 числа на луч (дальность, R, G, B) + one-hot направления.
RAY_CHANNELS = 4
NUM_MOVEMENT_ACTIONS = 4

# Шум только в момент наследования, до обучения за жизнь.
BIRTH_WEIGHT_NOISE_STD = 0.01

# Диапазоны включительные. step — шаг одной мутации.
STICKY_STEPS = (1, 20)
SWITCH_MARGIN = (0.0, 0.5, 0.02)
RANDOM_WALK_MIN_STEPS = (1, 20)
RANDOM_WALK_MAX_STEPS = (1, 30)
PAIN_REPEAT = (1, 8)
CYCLE_REPEATS = (2, 12)
LABEL_PENALTY = (0.0, 0.5, 0.02)
TRAIN_EVERY = (1, 12)
KEEP_FRACTION = (0.0, 1.0, 0.05)
INHERIT_WEIGHTS = (0.0, 1.0, 0.05)
