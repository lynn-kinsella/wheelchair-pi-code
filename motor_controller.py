import motor_utils

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


"""
Wait Time between updates sent to the PWM
"""
UPDATE_PERIOD = 0.01


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
Pin Definitions
"""
LPWM_pin = 35 #Left 
# LPWM_R = 40 
RPWM_pin = 12 #Right Motor
# RPWM_R = 38
# Not using reverse, all reverse setup commented

EN = 10

"""
Initialize Pins
"""
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BOARD)
GPIO.setup(RPWM_pin, GPIO.OUT, initial = GPIO.LOW)
GPIO.setup(EN, GPIO.OUT, initial = GPIO.LOW)
GPIO.setup(LPWM_pin, GPIO.OUT, initial = GPIO.LOW)
# GPIO.setup(LPWM_R, GPIO.OUT, initial = GPIO.LOW)
# GPIO.setup(RPWM_R, GPIO.OUT, initial = GPIO.LOW)

# Set pins as PWM
pwm_r = GPIO.PWM(RPWM_pin, 20000) #Right Motor
pwm_r.start(0)
# pwm1r = GPIO.PWM(38, 20000) #Right Motor, Reverse
# pwm1r.start(0)
pwm_l = GPIO.PWM(LPWM_pin, 20000) #Left Motor
pwm_l.start(0)
# pwm2r = GPIO.PWM(40, 20000) #Left Motor, Reverse
# pwm2r.start(0)


def set_PWM():
    output_enabled = False
    while True:
        # Get angle and speed info from external input
        speed, angle = PWM_queue.get()

        if speed < 5:
            pwm_r.ChangeDutyCycle(0)
            pwm_l.ChangeDutyCycle(0)
            output_enabled = False
            GPIO.output(EN, output_enabled)
        else:
            # Convert to PWM
            lpwm_new, rpwm_new = motor_utils.motor_map(speed, angle)

            # Enable motor output
            if output_enabled == False:
                output_enabled = True
                GPIO.output(EN, output_enabled)
                sleep(0.005)

            # Set left, right PWM
            pwm_r.ChangeDutyCycle(rpwm_new)
            pwm_l.ChangeDutyCycle(lpwm_new)

def dummy_input():
    while True:
        angle, speed = motor_utils.get_dummy_input()

        angle_input_queue.put(angle)
        speed_input_queue.put(speed)


def set_val_from_queue(old_val, q):
    try:
        return q.get(False)
    except Empty as e:
        return old_val
    
def update_speed_state(state):
    new_speed = state["target"]

    
    state["previous"] = state["current"]
    state["current"] = new_speed


def update_angle_state(state):    
    new_angle = state["target"]




    state["previous"] = state["current"]
    state["current"] = new_angle


def periodic_update():
    angle_state = {
        "current": 0,
        "previous": 0,
        "target": 0,
        "phase":0
    }
    speed_state = {
        "current": 0,
        "previous": 0,
        "target": 0,
        "phase":0
    }
    while True:
        angle_state["target"] = set_val_from_queue(angle_state["target"], angle_input_queue)
        speed_state["target"] = set_val_from_queue(speed_state["target"], speed_input_queue)

        update_speed_state(speed_state)
        update_angle_state(angle_state)

        PWM_queue.put((angle_state["current"], speed_state["current"]))

        sleep(UPDATE_PERIOD)

        

if __name__ == "__main__":
    pwm_thread = Thread(target=set_PWM)
    input_thread = Thread(target=dummy_input)
    state_thread = Thread(target=periodic_update)

    pwm_thread.start()
    input_thread.start()
    state_thread.start()