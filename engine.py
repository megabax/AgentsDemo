"""Движок: исполнительный механизм, двигающий агента (красный квадрат)."""

from player import Player


# Действия совместимы с Player.update
ACTION_UP = 0
ACTION_DOWN = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3
ACTION_STAY = 4

# Движение (для random/NN). STAY не используем в политике — иначе
# отрицательные примеры «не ходи X» размазывают вероятность на «стоять».
MOVEMENT_ACTIONS = (ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT)
ALL_ACTIONS = MOVEMENT_ACTIONS + (ACTION_STAY,)

# Противоположные направления (для анти-дребезга)
OPPOSITE_ACTIONS = {
    ACTION_UP: ACTION_DOWN,
    ACTION_DOWN: ACTION_UP,
    ACTION_LEFT: ACTION_RIGHT,
    ACTION_RIGHT: ACTION_LEFT,
}


class AgentEngine:
    """
    Применяет выбранное агентом действие к Player.
    Агент только решает «куда»; движок реально сдвигает красный квадрат.
    """

    def __init__(self, player: Player):
        self.player = player

    def bind(self, player: Player) -> None:
        """Привязать к игроку после reset()."""
        self.player = player

    def execute(self, action: int) -> bool:
        """
        Выполнить одно действие перемещения.
        Возвращает True при ударе о стену («боль»).
        """
        if action not in ALL_ACTIONS:
            raise ValueError(f"Неизвестное действие: {action}")
        return bool(self.player.update(action))
