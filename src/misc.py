import logging
from src.car_variables import CarType, HunterFeedbackCanIDs, KartFeedbackCanIDs
from src.gamepad import Gamepad
from src.gamepad.controller_mapping import ControllerMapping
from src.can_interface.can_decoder import print_can_messages, broadcast_can_message

def calculate_throttle(controller_axis_value: float, car_type: CarType) -> float | None:
    if car_type == CarType.kart:
        x = max(0.0, -controller_axis_value)
        return (x ** 3) * 100
    elif car_type == CarType.hunter:
        return -(controller_axis_value ** 3) * 1500

def calculate_steering(controller_axis_value: float, car_type: CarType) -> float | None:
    cubic = -(controller_axis_value ** 3)
    if car_type == CarType.kart:
        return -cubic * 1.25
    elif car_type == CarType.hunter:
        return cubic * 576

def dead_man_switch(gamepad: Gamepad) -> bool:
    if gamepad.axis(ControllerMapping.park) == 1:
        park = True
    else:
        park = False
    return park

def controller_break_value(gamepad: Gamepad) -> int:
    return max(0,round((gamepad.axis(ControllerMapping.park)**3)*100))

def setup_listeners(can_controller, car_type):
    """Register CAN message listeners."""
    if car_type == CarType.hunter:
        for feedback_id in HunterFeedbackCanIDs:
            can_controller.add_listener(feedback_id, print_can_messages)
            can_controller.add_listener(feedback_id, broadcast_can_message)
            logging.debug(f"Listener added for Hunter message ID: {feedback_id}")
    elif car_type == CarType.kart:
        for feedback_id in KartFeedbackCanIDs:
            can_controller.add_listener(feedback_id, print_can_messages)
            can_controller.add_listener(feedback_id, broadcast_can_message)
            logging.debug(f"Listener added for Kart message ID: {feedback_id}")