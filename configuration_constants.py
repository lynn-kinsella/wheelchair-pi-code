"""
PIN NUMBERS
"""
LEFT_PWM_PIN = 35
LEFT_REVERSE_PWM_PIN = 40
RIGHT_PWM_PIN = 12
RIGHT_REVERSE_PWM_PIN = 38

MOTOR_ENABLE_PIN = 10

"""
DELAYS AND SLEEPS
"""
MOTOR_SLEEP_TIME = 0.01
UPDATE_PERIOD = 0.01

"""
PROTOCOL CONSTANTS
"""
OSC_SERVER_IP = "0.0.0.0"
OSC_SERVER_PORT = 5000


"""
OSC PARAMETERS
"""
BCI_HISTORY_DEQUE_LENGTH = 30

"""
CONTROL SYSTEM CONSTANTS
"""
UPDATE_PERIOD = 0.01
SPEED_PWM_DEADZONE = 5
MOTOR_PWM_FREQUENCY = 1000
ACCELERATION_STEP = 0.1
ANGLE_DIFF_MULTIPLIER = 0.5
