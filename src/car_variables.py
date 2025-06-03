from enum import IntEnum

CAN_MESSAGE_SENDING_SPEED = 0.04

class HunterControlMode(IntEnum):
    idle_mode = 0
    command_mode = 1

class HunterControlCanIDs(IntEnum):
    movement_control = 0x111
    control_mode = 0x421
    parking_control = 0x131

class HunterFeedbackCanIDs(IntEnum):
    movement_feedback = 0x221
    status_feedback = 0x211

class KartControlCanIDs(IntEnum):
    breaking = 0x110
    steering = 0x220
    throttle = 0x330

class KartFeedbackCanIDs(IntEnum):
    steering_sensor = 0x1E5
    breaking_sensor = 0x710
    speed_sensor = 0x440
    internal_throttle = 0x730

class KartGearBox(IntEnum):
    neutral = 0
    forward = 1
    backward = 2

class CarType(IntEnum):
    kart = 0
    hunter = 1

class ControlMode(IntEnum):
    automatic = 0
    manual = 1

class CameraResolution(IntEnum):
    WIDTH = 848
    HEIGHT = 480

class LineDetectionDims(IntEnum):
    WIDTH = 720
    HEIGHT = 720