import random
import motor_utils
from configuration_constants import *
from state_enums import SpeedStates, AngleStates, AnglePred

from threading import Thread, Lock, Event
from multiprocessing import Manager, Process
from queue import Queue, Empty
from collections import deque
import RPi.GPIO as GPIO
from time import sleep
import os
from pythonosc import osc_server, dispatcher
from time import time
from collections import deque

import tensorflow as tf
import numpy as np
import mediapipe as mp
import cv2 as cv

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


def set_PWM(PWM_queue):
    try:
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
                # print(lpwm_new, rpwm_new)
                # Set left, right PWM
                pwm_r.ChangeDutyCycle(rpwm_new)
                pwm_l.ChangeDutyCycle(lpwm_new)
                sleep(MOTOR_SLEEP_TIME)
            # print(pwm_r, " -- ", pwm_l)
    except:
        pwm_r.ChangeDutyCycle(0)
        pwm_l.ChangeDutyCycle(0)
        output_enabled = False
        GPIO.output(MOTOR_ENABLE_PIN, output_enabled)




def dummy_input(speed_input_queue, angle_input_queue):
    i = 0
    while True:
        i = (i+1)%2
        speed, angle = i, i
        #speed, angle = motor_utils.dummy_external_input()

        angle_input_queue.put(angle)
        speed_input_queue.put(speed)
        sleep(1)


def eeg_handler(address: str, fixed_args: list, *args):
    shared_buffer = fixed_args[0]
    if len(args) == 4: 
        if len(shared_buffer) == BCI_PREDICT_WINDOW+5:
            del shared_buffer[0]
        shared_buffer.append(args)


def osc_server_handler(shared_buffer):
    osc_dispatcher = dispatcher.Dispatcher()
    osc_dispatcher.map("/muse/eeg", eeg_handler, shared_buffer)

    # server = osc_server.ThreadingOSCUDPServer((ip, port), dispatcher)
    server = osc_server.BlockingOSCUDPServer((OSC_SERVER_IP, OSC_SERVER_PORT), osc_dispatcher)
    print("Listening on UDP port "+str(OSC_SERVER_PORT))
    server.serve_forever()


def prediction_server(shared_buffer, speed_input_queue):
    BCI_history = deque(maxlen=BCI_SMOOTHING_WINDOW)
    tf_model = tf.keras.models.load_model(MODEL_DIR)
    while True:
        if len(shared_buffer) < BCI_PREDICT_WINDOW:
            continue
        else:
            data = np.array([shared_buffer[:BCI_PREDICT_WINDOW]]).transpose(0, 2, 1)
            prediction = np.argmax(tf_model.predict(data, verbose=0)[0])

            BCI_history.appendleft(prediction)

            if prediction == SpeedStates.DISCONNECTED.value:
                for i in range(BCI_SMOOTHING_WINDOW):
                    BCI_history.appendleft(prediction)

            hist_weighted_prediction = max(BCI_history.queue, key=BCI_history.queue.count)

            # if hist_weighted_prediction == SpeedStates.DECCEL.value:
            #     for i in range(BCI_SMOOTHING_WINDOW):
            #         BCI_history.put(hist_weighted_prediction)

            speed_pred = hist_weighted_prediction
            
            speed_input_queue.put(speed_pred)


def tcp_receiver(video_frame_queue):
    print("Connecting to tcp video stream")
    # cap = cv.VideoCapture("tcp://MM@172.20.10.6:3333")
    cap = cv.VideoCapture('tcp://MM.local:3333')
    while True:
        ret, frame = cap.read()
        if not ret:
            # print("Error reading frame")
            sleep(1/10/2) # Sleep for half a frame @ 10 FPS
            continue
        frame = cv.flip(frame, 1)
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        if video_frame_queue.full():
            video_frame_queue.get()    
        video_frame_queue.put(rgb_frame)


def eye_tracking(video_frame_queue, angle_input_queue):    
    eye_tracking_history = Queue(maxsize=EYE_SMOOTHING_WINDOW)
    
    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:
        while True:
            frame = video_frame_queue.get()
            img_h, img_w = frame.shape[:2]
            results = face_mesh.process(frame)

            if results.multi_face_landmarks:
                # Get Mesh Points
                mesh_points=np.array([np.multiply([p.x, p.y], [img_w, img_h]).astype(int) for p in results.multi_face_landmarks[0].landmark])
                
                left_iris_points = mesh_points[LEFT_IRIS]
                right_iris_points = mesh_points[RIGHT_IRIS]

                left_iris = left_iris_points[0][0] + left_iris_points[1][0] + left_iris_points[2][0] + left_iris_points[3][0]
                left_iris = left_iris//4
                right_iris = right_iris_points[0][0] + right_iris_points[1][0] + right_iris_points[2][0] + right_iris_points[3][0]
                right_iris = right_iris//4

                ### Calculate Eye Corners
                lcorners = mesh_points[LEFT_CORNERS]
                lcorners = [lcorners[0][0], lcorners[1][0]]
                rcorners = mesh_points[RIGHT_CORNERS]
                rcorners = [rcorners[0][0], rcorners[1][0]]

                # Calculate angle
                # Represented as % of distance the centre of iris is from left corner to right corner of eye
                total_dist = lcorners[1] - lcorners[0] + rcorners[1] - rcorners[0]
                offset = left_iris - lcorners[0] + right_iris - rcorners[0]
                angle = offset/total_dist

                if angle < LEFT_CENTER_DEADZONE and angle > RIGHT_CENTER_DEADZONE:
                    angle = 0
                elif angle <= RIGHT_CENTER_DEADZONE:
                    angle = MAX_ANGLE
                elif angle >= LEFT_CENTER_DEADZONE:
                    angle = -MAX_ANGLE
                        
                # using queue instead
                if eye_tracking_history.full():
                    eye_tracking_history.get()
                eye_tracking_history.put(angle)

                hist_weighted_prediction = max(eye_tracking_history.queue, key=eye_tracking_history.queue.count)
                #if prev_weighted_prediction != hist_weighted_prediction:
                #    for i in range(len(BCI_history)//2):
                #        BCI_history.pop()
                angle_pred = hist_weighted_prediction

                # print("Eye Tracking: ", AnglePred( (int)(angle_pred)).name)
                angle_input_queue.put(angle_pred)
                # sleep(0.1)


def set_val_from_queue(old_val, q):
    if not q.empty():
        val = q.get(False)
        if val == SpeedStates.DISCONNECTED:
            val = old_val
        return val
    else:
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
            if state["speed"] == PEAK_PWM_SPEED:
                new_acceleration = 0
    
    state["accel"] = new_acceleration

    new_speed = state["speed"] + state["accel"]
    new_speed = min(PEAK_PWM_SPEED, new_speed)
    new_speed = max(0, new_speed)

    state["speed"] = new_speed
    return state

 
def update_angle_state(state):   
    
    diff = state["target"] - state["current"]
    step = 0
    if abs(diff) > 0:
        step = ANGLE_STEP_SIZE
        step = min(abs(diff), step) 
        step *= diff/abs(diff) 

    state["current"] = state["current"] + step
    return state


def periodic_update(PWM_queue, angle_input_queue, speed_input_queue):
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
        
        if speed_state["phase"] == SpeedStates.DISCONNECTED:
            speed_state["phase"] = SpeedStates.DECCEL

        speed_state  = update_speed_state(speed_state)
        angle_state = update_angle_state(angle_state)

        PWM_queue.put((angle_state["current"], speed_state["speed"]))
        # print(angle_state['current'], speed_state['speed'])

        sleep(UPDATE_PERIOD)
        

if __name__ == "__main__":
    # Buffers 

    
    # Buffers 

    speed_input_queue = Manager().Queue()
    angle_input_queue = Manager().Queue()
    PWM_queue = Manager().Queue()
    video_frame_queue = Manager().Queue(3)
    
    # OSC Server internal synchronization variables
    # FIXME: Using a list here could be slower, we'll have to see when running pi
    # Solution can be to put osc and prediction server thread on the same process
    shared_buffer = Manager().list()

    process_list = []

    if os.environ["INPUT_MODE"] == "LIVE":
        osc_process = Process(target=osc_server_handler, args=(shared_buffer,))
        process_list.append(osc_process)
        prediction_process = Process(target=prediction_server, args=(shared_buffer, speed_input_queue))
        process_list.append(prediction_process)

        tcp_rx_thread = Process(target=tcp_receiver, args=(video_frame_queue,))
        process_list.append(tcp_rx_thread)
        eye_tracking_thread = Process(target=eye_tracking, args=(video_frame_queue, angle_input_queue))
        process_list.append(eye_tracking_thread)

    else:
        dummy_process = Process(target=dummy_input, args=(speed_input_queue, angle_input_queue))
        process_list.append(dummy_process)

    state_process = Process(target=periodic_update, args=(PWM_queue, angle_input_queue, speed_input_queue))
    process_list.append(state_process)

    for process in process_list:
        process.start()

    set_PWM(PWM_queue)

    for process in process_list:
        process.join()
