from enum import StrEnum

class ControllerMapping(StrEnum):
    buttonExit = 'B'
    L_joystickX = 'LEFT-X'
    L_joystickY = 'LEFT-Y'
    R_joystickX = 'RIGHT-X'
    R_joystickY = 'RIGHT-Y'
    throttle = 'LT'
    park = 'RT'