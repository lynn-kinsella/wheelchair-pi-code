from enum import Enum
class SpeedStates (Enum):
    DECCEL = 0
    ACCEL = 1
    REST = 4
    DISCONNECTED = 5

class AngleStates (Enum):
    REST = 0
    INCREASE_FROM_POSITIVE = 1
    INCREASE_FROM_NEGATIVE = 2
    DECREASE_FROM_NEGATIVE = -1
    DECREASE_FROM_POSITIVE = -2

class AnglePred (Enum):
    CENTER = 0
    LEFT = -50
    RIGHT = 50