import math

def motor_map(angle, speed):
    """
    Convert an angle in radians, with pi/2 rad at fully right
    and -pi/2 rad at fully left and a scalar speed to a motor PWMs
    """
    
    if (angle >= 0 ):
        left_factor = 1 - abs(2*angle/math.pi)
        right_factor = 1
    if (angle < 0):
        left_factor = 1
        right_factor = 1 - abs(2*angle/math.pi)

    left_PWM = left_factor*speed
    right_PWM = right_factor*speed

    return (int(left_PWM), int(right_PWM))

def dummy_external_input():
    angle = input("Enter angle (rads, -1.57 to 1.57): ")
    speed = input("Enter speed (0-100): ")
    return (int(speed), float(angle))


def test():
    angle = input("Enter angle (rads, -1.57 to 1.57): ")
    speed = input("Enter speed (0-100): ")
    print(motor_map(float(angle), float(speed)))

if __name__ == "__main__":
    test()