import motor_utils
from configuration_constants import *
from state_enums import SpeedStates, AngleStates 

from threading import Thread, Lock, Event
from queue import Queue, Empty
import RPi.GPIO as GPIO
from time import sleep


"""
Shared Variables
"""
speed_input_queue = Queue()
angle_input_queue = Queue()
PWM_queue = Queue()

'''
Pin List - 3/8/23
4 - 5V high
6 - GND
10 - Enable (for motors, disable for break)
12 - Right Motor FWD PWM
38 -  Right Motor REV PWM
40 - Left Motor REV PWM
35 - Left Motor FWD PWM
'''

"""
Initialize Pins
"""
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BOARD)
GPIO.setup(RIGHT_PWM_PIN, GPIO.OUT, initial = GPIO.LOW)
GPIO.setup(MOTOR_ENABLE_PIN, GPIO.OUT, initial = GPIO.LOW)
GPIO.setup(LEFT_PWM_PIN, GPIO.OUT, initial = GPIO.LOW)
# GPIO.setup(LPWM_R, GPIO.OUT, initial = GPIO.LOW)
# GPIO.setup(RPWM_R, GPIO.OUT, initial = GPIO.LOW)

# Set pins as PWM
pwm_r = GPIO.PWM(RIGHT_PWM_PIN, MOTOR_PWM_FREQUENCY) #Right Motor
pwm_r.start(0)
# pwm1r = GPIO.PWM(38, 20000) #Right Motor, Reverse
# pwm1r.start(0)
pwm_l = GPIO.PWM(LEFT_PWM_PIN, MOTOR_PWM_FREQUENCY) #Left Motor
pwm_l.start(0)
# pwm2r = GPIO.PWM(40, 20000) #Left Motor, Reverse
# pwm2r.start(0)


def set_PWM():
    output_enabled = False
    while True:
        # Get angle and speed info from external input
        angle, speed = PWM_queue.get()

        if speed < SPEED_PWM_DEADZONE:
            pwm_r.ChangeDutyCycle(0)
            pwm_l.ChangeDutyCycle(0)
            output_enabled = False
            GPIO.output(MOTOR_ENABLE_PIN, output_enabled)
        else:
            # Convert to PWM
            lpwm_new, rpwm_new = motor_utils.motor_map(angle, speed)

            # Enable motor output
            if output_enabled == False:
                output_enabled = True
                GPIO.output(MOTOR_ENABLE_PIN, output_enabled)
                sleep(MOTOR_SLEEP_TIME)

            # Set left, right PWM
            pwm_r.ChangeDutyCycle(rpwm_new)
            pwm_l.ChangeDutyCycle(lpwm_new)
            sleep(MOTOR_SLEEP_TIME)

def dummy_input():
    while True:
        speed, angle = motor_utils.dummy_external_input()

        angle_input_queue.put(angle)
        speed_input_queue.put(speed)


def set_val_from_queue(old_val, q):
    try:
        return q.get(False)
    except Empty as e:
        return old_val
    
def update_speed_state(state):
    new_acceleration = 0
    if state["phase"] == SpeedStates.DECCEL or (state["phase"] == SpeedStates.REST and state["accel"] > 0):
        new_acceleration = state["accel"] - ACCELERATION_STEP
        if state["phase"] == SpeedStates.REST:
            new_acceleration = min(0, new_acceleration)
        else:
            if state["speed"] == 0:
                new_acceleration = 0

    elif state["phase"] == SpeedStates.ACCEL or (state["phase"] == SpeedStates.REST and state["accel"] < 0):
        new_acceleration = state["accel"] + ACCELERATION_STEP
        if state["phase"] == SpeedStates.REST:
            new_acceleration = max(0, new_acceleration)
        else:
            if state["speed"] == 100:
                new_acceleration = 0
    
    state["accel"] = new_acceleration

    new_speed = state["speed"] + state["accel"]
    new_speed = min(100, new_speed)
    new_speed = max(0, new_speed)

    state["speed"] = new_speed
    return state


def update_angle_state(state):   
    diff = state["target"] - state["current"]
    new_angle = state["current"] + diff*ANGLE_DIFF_MULTIPLIER

    state["previous"] = state["current"]
    state["current"] = new_angle
    return state


def periodic_update():
    angle_state = {
        "current": 0,
        "previous": 0,
        "target": 0,
        "phase": AngleStates.REST
    }
    speed_state = {
        "speed": 0,
        "accel": 0,
        "phase": SpeedStates.REST
    }
    while True:
        angle_state["target"] = set_val_from_queue(angle_state["target"], angle_input_queue)
        speed_state["phase"] = SpeedStates(set_val_from_queue(speed_state["phase"], speed_input_queue))

        speed_state  = update_speed_state(speed_state)
        angle_state = update_angle_state(angle_state)

        PWM_queue.put((angle_state["current"], speed_state["speed"]))

        sleep(UPDATE_PERIOD)

        

if __name__ == "__main__":
    pwm_thread = Thread(target=set_PWM)
    input_thread = Thread(target=dummy_input)
    state_thread = Thread(target=periodic_update)

    pwm_thread.start()
    input_thread.start()
    state_thread.start()