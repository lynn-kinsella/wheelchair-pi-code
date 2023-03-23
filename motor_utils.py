import math

def motor_map(angle, speed):
    """
    Convert an angle in % from -100 being fully right and 100 being fully left
    """
    
    if (angle <= 0 ):
        # left_factor = 1 - abs(2*angle/math.pi)
        # right_factor = 1
        left_pwm = speed - angle
        right_pwm = speed
    if (angle > 0):
        right_pwm = speed + angle
        left_pwm = speed
        # left_factor = 1
        # right_factor = 1 - abs(2*angle/math.pi)

    # left_PWM = left_factor*speed
    # right_PWM = right_factor*speed

    return (int(left_pwm), int(right_pwm))

def dummy_external_input():
    angle = input("Enter angle (-100 to 100): ")
    speed = input("Enter acceleration (1, 0, -1): ")
    return (int(speed), float(angle))


def test():
    angle = input("Enter angle (-100 to 100): ")
    speed = input("Enter speed (0, 100): ")
    print(motor_map(float(angle), int(speed)))

if __name__ == "__main__":
    test()