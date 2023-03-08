import motor_utils

import RPi.GPIO as GPIO
from time import sleep

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

LPWM_pin = 35 #Left 
# LPWM_R = 40 
RPWM_pin = 12 #Right Motor
# RPWM_R = 38
# Not using reverse, all reverse setup commented

EN = 10

# Initialize Pins
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
    while True:
        # Get angle and speed info from external input
        speed, angle = motor_utils.dummy_external_input()

        if speed < 5:
            pwm_r.ChangeDutyCycle(0)
            pwm_l.ChangeDutyCycle(0)
            GPIO.output(EN, False)
        else:
            # Convert to PWM
            lpwm_new, rpwm_new = motor_utils.motor_map(speed, angle)

            # Enable motor output
            GPIO.ouput(EN, True)

            # Set left, right PWM
            pwm_r.ChangeDutyCycle(rpwm_new)
            pwm_l.ChangeDutyCycle(lpwm_new)



if __name__ == "__main__":
    print()
