import motor_utils
from configuration_constants import *
from state_enums import SpeedStates, AngleStates 

from threading import Thread, Lock, Event
from queue import Queue, Empty
from time import sleep

from pythonosc import osc_server, dispatcher

import os
import numpy as np
import tensorflow as tf

class Motor:
    def __init__(self, dc=0):
        self.pwm = dc
    def ChangeDutyCycle(self, dc):
        self.pwm = dc
    def __str__(self) -> str:
        return str(self.pwm)

class GPIO():
    def output(pin, state):
        #print("Pin ", pin, " in state ", state)
        pass

pwm_l = Motor()
pwm_r = Motor()

"""
Shared Variables
"""
speed_input_queue = Queue()
angle_input_queue = Queue()
PWM_queue = Queue()

# OSC Server internal synchronization variables
shared_buffer = Queue(maxsize=PREDICT_WINDOW)

if os.environ["INPUT_MODE"] == "LIVE":
    tf_model = tf.keras.models.load_model("./model_saved")
else:
    tf_model = None

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
        # print(pwm_r, " -- ", pwm_l)


def dummy_input():
    while True:
        speed, angle = motor_utils.dummy_external_input()

        angle_input_queue.put(angle)
        speed_input_queue.put(speed)


def eeg_handler(address: str,*args):
    if len(args) == 4: 
        if shared_buffer.full():
            shared_buffer.get()
        shared_buffer.put(args)


def osc_server_handler():
    osc_dispatcher = dispatcher.Dispatcher()
    osc_dispatcher.map("/muse/eeg", eeg_handler)

    # server = osc_server.ThreadingOSCUDPServer((ip, port), dispatcher)
    server = osc_server.BlockingOSCUDPServer((OSC_SERVER_IP, OSC_SERVER_PORT), osc_dispatcher)
    print("Listening on UDP port "+str(OSC_SERVER_PORT))
    server.serve_forever()


def prediction_server():
    BCI_history = Queue(maxsize=SMOOTHING_WINDOW)
    prev_weighted_prediction = 0
    while True:
        if not shared_buffer.full():
            continue
        else:
            data = np.array([shared_buffer.queue]).transpose(0, 2, 1)
            prediction = np.argmax(tf_model.predict(data, verbose=0)[0])

            # using queue instead
            if BCI_history.full():
                BCI_history.get()
            BCI_history.put(prediction)

            hist_weighted_prediction = max(BCI_history.queue, key=BCI_history.queue.count)
            #if prev_weighted_prediction != hist_weighted_prediction:
            #    for i in range(len(BCI_history)//2):
            #        BCI_history.pop()

            prev_weighted_prediction = hist_weighted_prediction
            speed_pred = hist_weighted_prediction
            speed_input_queue.put(speed_pred)
            print(speed_pred)

        #angle, dummy = motor_utils.dummy_external_input()                                  
        angle = 0
        angle_input_queue.put(angle)


def set_val_from_queue(old_val, q):
    try:
        val = q.get(False)
        if val == SpeedStates.DISCONNECTED:
            val = old_val
        return val
    except Empty as e:
        return old_val
    

def update_speed_state(state):
    new_acceleration = 0
    if state["phase"] == SpeedStates.DECCEL or (state["phase"] == SpeedStates.REST and state["accel"] > 0):
        new_acceleration = state["accel"] - DECCELERATION_STEP
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
    thread_list = []
    pwm_thread = Thread(target=set_PWM)
    thread_list.append(pwm_thread)

    if os.environ["INPUT_MODE"] == "LIVE":
        osc_thread = Thread(target=osc_server_handler)
        thread_list.append(osc_thread)
        prediction_thread = Thread(target=prediction_server)
        thread_list.append(prediction_thread)

    else:
        dummy_thread = Thread(target=dummy_input)
        thread_list.append(dummy_thread)

    state_thread = Thread(target=periodic_update)
    thread_list.append(state_thread)

    for thread in thread_list:
        thread.start()