from enum import Enum
class SpeedStates (Enum):
    REST = 0
    ACCEL = 1
    DECCEL = -1

class AngleStates (Enum):
    REST = 0
    INCREASE_FROM_POSITIVE = 1
    INCREASE_FROM_NEGATIVE = 2
    DECREASE_FROM_NEGATIVE = -1
    DECREASE_FROM_POSITIVE = -2