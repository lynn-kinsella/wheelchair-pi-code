from evdev import InputDevice, categorize, ecodes
from select import select
import RPi.GPIO as GPIO
import os
import sys
from time import sleep
LPWM = 35 #Left Motor
REN = 10
RPWM = 12 #Right Motor
LEN = 16
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BOARD)
GPIO.setup(RPWM, GPIO.OUT, initial = GPIO.LOW)
GPIO.setup(REN, GPIO.OUT, initial = GPIO.LOW)
GPIO.setup(LPWM, GPIO.OUT, initial = GPIO.LOW)
GPIO.setup(LEN, GPIO.OUT, initial = GPIO.LOW)
GPIO.setup(40, GPIO.OUT, initial = GPIO.LOW)
GPIO.setup(38, GPIO.OUT, initial = GPIO.LOW)

gamepad = InputDevice('/dev/input/event0')
gamepad.capabilities()
pwm1 = GPIO.PWM(RPWM, 20000) #Right Motor
pwm1.start(0)
pwm1r = GPIO.PWM(38, 20000) #Right Motor, Reverse
pwm1r.start(0)
pwm2 = GPIO.PWM(LPWM, 20000) #Left Motor
pwm2.start(0)
pwm2r = GPIO.PWM(40, 20000) #Left Motor, Reverse
pwm2r.start(0)
    
while True:
   
    r,w,x = select([gamepad], [], [])
    for event in gamepad.read():
        if event.type == ecodes.EV_KEY:
            keyevent = categorize(event)
            if keyevent.scancode == 304:
                print('Back')
            elif keyevent.scancode == 305:
                print('Right')
                
                if keyevent.keystate == 1:
                    pwm2.ChangeDutyCycle(0)
                    pwm1r.ChangeDutyCycle(0)
                    GPIO.output(REN, True) # Always set before driving motors
                    d = 10 # duty cycle
                    
                    for i in range (8):
                        pwm1.ChangeDutyCycle(d)
                        
                        pwm2r.ChangeDutyCycle(d)
                        d = d + 10 # increase speed over time
                        sleep(0.1)
                        
                elif keyevent.keystate == 0:
                    d = 0
                    for i in range (10):
                        d = d + 10
                        pwm1.ChangeDutyCycle(100-d)
                        pwm2r.ChangeDutyCycle(100-d)
                        sleep(0.1)
                    GPIO.output(REN, False) # Set after no longer running motors
                    
                   
            elif keyevent.scancode == 308:
                print('Forward')
                
                if keyevent.keystate == 1:
                    pwm1r.ChangeDutyCycle(0)
                    pwm2r.ChangeDutyCycle(0)
                    GPIO.output(REN, True)
                    
                    d = 10
                    for i in range (8):
                        pwm1.ChangeDutyCycle(d)
                        pwm2.ChangeDutyCycle(d)
                        d = d + 10
                        sleep(0.1)
                        
                elif keyevent.keystate == 0:
                    d = 0
                    for i in range (10):
                        d = d + 10
                        pwm1.ChangeDutyCycle(100-d)
                        pwm2.ChangeDutyCycle(100-d)
                        sleep(0.1)
                    GPIO.output(REN, False)
                   
            elif keyevent.scancode == 307: 
                print('Left')
                if keyevent.keystate == 1:
                    pwm1.ChangeDutyCycle(0)
                    pwm2r.ChangeDutyCycle(0)
                    GPIO.output(REN, True)
                    d = 10
                    for i in range (8):                        
                        pwm2.ChangeDutyCycle(d)
                        pwm1r.ChangeDutyCycle(d)
                        d = d + 10
                        sleep(0.1)
                        
                elif keyevent.keystate == 0:
                    d = 0
                    for i in range (10):
                        d = d + 10
                        pwm2.ChangeDutyCycle(100-d)
                        pwm1r.ChangeDutyCycle(100-d)
                        sleep(0.1)
                    GPIO.output(REN, False)


