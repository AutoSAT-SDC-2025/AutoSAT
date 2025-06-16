"""
Miscellaneous utility functions for AutoSAT vehicle control.

Provides gamepad input processing, vehicle-specific control calculations,
and CAN message listener setup for both Hunter and Kart vehicles.
"""

import logging
from src.car_variables import CarType, HunterFeedbackCanIDs, KartFeedbackCanIDs, HunterControlCanIDs, KartControlCanIDs
from src.gamepad import Gamepad
from src.gamepad.controller_mapping import ControllerMapping
from src.can_interface.can_decoder import print_can_messages, broadcast_can_message

def calculate_throttle(controller_axis_value: float, car_type: CarType) -> float | None:
    """
    Convert gamepad axis input to vehicle-specific throttle command.
    
    Applies cubic response curve for smooth control and scales to appropriate
    range for each vehicle type.
    
    Args:
        controller_axis_value: Raw gamepad axis value (-1.0 to 1.0)
        car_type: Vehicle type for appropriate scaling
        
    Returns:
        Throttle command value:
        - Kart: 0-100 (percentage)
        - Hunter: -1500 to 1500
        - None: if unsupported vehicle type
    """
    if car_type == CarType.kart:
        # Kart: Only forward throttle (0-100%), cubic curve for smoothness
        x = max(0.0, -controller_axis_value)
        return (x ** 3) * 100
    elif car_type == CarType.hunter:
        # Hunter: Bidirectional speed control, cubic curve
        return -(controller_axis_value ** 3) * 1500

def calculate_steering(controller_axis_value: float, car_type: CarType) -> float | None:
    """
    Convert gamepad axis input to vehicle-specific steering command.
    
    Applies cubic response curve for fine control near center and scales
    to each vehicle's steering range.
    
    Args:
        controller_axis_value: Raw gamepad axis value (-1.0 to 1.0)
        car_type: Vehicle type for appropriate scaling
        
    Returns:
        Steering command value:
        - Kart: ±1.25 radians (physical steering limit)
        - Hunter: ±576(manufacturer specification)
        - None: if unsupported vehicle type
    """
    cubic = -(controller_axis_value ** 3)
    if car_type == CarType.kart:
        # Kart: Physical steering angle in radians
        return -cubic * 1.25
    elif car_type == CarType.hunter:
        # Hunter: CAN protocol steering units
        return cubic * 576

def dead_man_switch(gamepad: Gamepad) -> bool:
    """
    More of a break pedal than a dead man switch. Should be renamed.
    
    Args:
        gamepad: Xbox gamepad instance
        
    Returns:
        True if trigger is fully pressed (trigger pulled)
    """
    if gamepad.axis(ControllerMapping.park) == 1:
        park = True
    else:
        park = False
    return park

def controller_break_value(gamepad: Gamepad) -> int:
    """
    Calculate brake intensity from gamepad trigger input.
    
    Converts analog trigger position to brake command with cubic response
    for progressive braking control.
    
    Args:
        gamepad: Xbox gamepad instance
        
    Returns:
        Brake value (0-100) where 0 is no braking, 100 is maximum brake
    """
    return max(0, round((gamepad.axis(ControllerMapping.park)**3)*100))

def setup_listeners(can_controller, car_type):
    """
    Register CAN message listeners for vehicle feedback and control monitoring.
    
    Sets up message handlers for both console logging and web interface
    broadcasting based on vehicle type.
    
    Args:
        can_controller: Vehicle CAN controller instance
        car_type: Vehicle type (CarType.hunter or CarType.kart)
    """
    if car_type == CarType.hunter:
        # Register listeners for Hunter feedback messages
        for feedback_id in HunterFeedbackCanIDs:
            can_controller.add_listener(feedback_id, print_can_messages)
            can_controller.add_listener(feedback_id, broadcast_can_message)
            logging.debug(f"Listener added for Hunter feedback message ID: {feedback_id}")
        
        # Register listeners for Hunter control command echo
        for control_id in HunterControlCanIDs:
            can_controller.add_listener(control_id, print_can_messages)
            can_controller.add_listener(control_id, broadcast_can_message)
            logging.debug(f"Listener added for Hunter control message ID: {control_id}")
            
    elif car_type == CarType.kart:
        # Register listeners for Kart feedback messages
        for feedback_id in KartFeedbackCanIDs:
            can_controller.add_listener(feedback_id, print_can_messages)
            can_controller.add_listener(feedback_id, broadcast_can_message)
            logging.debug(f"Listener added for Kart feedback message ID: {feedback_id}")
        
        # Register listeners for Kart control command echo
        for control_id in KartControlCanIDs:
            can_controller.add_listener(control_id, print_can_messages)
            can_controller.add_listener(control_id, broadcast_can_message)
            logging.debug(f"Listener added for Kart control message ID: {control_id}")
