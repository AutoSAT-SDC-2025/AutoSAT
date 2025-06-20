"""
Hunter vehicle CAN controller implementation.

Provides CAN bus communication interface for Hunter autonomous vehicles,
handling movement control, parking brake, and control mode switching.
"""

import logging
import struct
import threading
import can
from src.can_interface.can_controller_interface import ICanController, init_can_message
from src.car_variables import HunterControlCanIDs, CAN_MESSAGE_SENDING_SPEED, HunterFeedbackCanIDs, HunterControlMode

class HunterCanController(ICanController):
    """
    CAN controller implementation for Hunter vehicles.
    
    Manages periodic message transmission for movement control, parking brake,
    and control mode. Supports combined steering/throttle commands and provides
    listener interface for feedback messages.
    """

    can_bus: can.Bus
    __listeners: dict[int, list[callable]]
    __thread: threading.Thread
    __running: bool
    
    def __init__(self, bus: can.Bus) -> None:
        """
        Initialize Hunter CAN controller with periodic message tasks.
        
        Args:
            bus: CAN bus interface for communication
        """
        self.can_bus = bus
        self.__listeners = {}

        self.__movement_control_message = init_can_message(HunterControlCanIDs.movement_control.value)
        self.__movement_control_task = self.can_bus.send_periodic(self.__movement_control_message,
                                                                  CAN_MESSAGE_SENDING_SPEED)

        self.__parking_control_message = init_can_message(HunterControlCanIDs.parking_control.value)
        self.__parking_control_task = self.can_bus.send_periodic(self.__parking_control_message,
                                                                 CAN_MESSAGE_SENDING_SPEED)

        self.__control_mode_message = init_can_message(HunterControlCanIDs.control_mode.value)
        self.__control_mode_task = self.can_bus.send_periodic(self.__control_mode_message, CAN_MESSAGE_SENDING_SPEED)
        self.__running = False

    def add_listener(self, message_id: HunterFeedbackCanIDs, listener: callable) -> None:
        """
        Register listener for specific feedback message ID.
        
        Args:
            message_id: Hunter feedback message ID to monitor
            listener: Callback function for received messages
        """
        if message_id not in self.__listeners:
            self.__listeners[message_id] = []
        self.__listeners[message_id].append(listener)

    def start(self) -> None:
        """Start message listening thread."""
        self.__running = True
        self.__thread = threading.Thread(target=self.__listen, daemon=True)
        self.__thread.start()
        logging.debug("started thread for __listen")

    def set_steering(self, steering_angle: float) -> None:
        """Hunter uses combined steering/throttle commands - no separate steering."""
        pass

    def set_throttle(self, throttle_value: float) -> None:
        """Hunter uses combined steering/throttle commands - no separate throttle."""
        pass

    def set_kart_gearbox(self, kart_gearbox) -> None:
        """Hunter vehicles don't have gearbox control."""
        pass

    def set_break(self, break_value: int) -> None:
        """Hunter uses parking brake instead of proportional braking."""
        pass

    def set_steering_and_throttle(self, steering_angle: float, throttle_value: float) -> None:
        """
        Set combined steering and throttle command for Hunter vehicles.
        
        Args:
            steering_angle: Steering angle
            throttle_value: Speed value
        """
        steering_angle_bytes = struct.pack('>h', int(steering_angle))
        speed_bytes = struct.pack('>h', int(throttle_value))
        self.__movement_control_message.data = list(bytearray(speed_bytes)) + [0, 0, 0, 0] + list(bytearray(steering_angle_bytes))
        self.__movement_control_task.modify_data(self.__movement_control_message)

    def set_control_mode(self, control_mode: HunterControlMode) -> None:
        """
        Set Hunter vehicle control mode.
        
        Args:
            control_mode: Control mode (idle, command, remote)
        """
        self.__control_mode_message.data = list(bytearray(struct.pack('?', control_mode.value)))
        self.__control_mode_task.modify_data(self.__control_mode_message)

    def set_parking_mode(self, parking_value: bool):
        """
        Control Hunter parking brake.
        
        Args:
            parking_value: True to engage, False to disengage parking brake
        """
        self.__parking_control_message.data = list(bytearray(struct.pack('?', parking_value)))
        self.__parking_control_task.modify_data(self.__parking_control_message)

    def stop(self) -> None:
        """Stop CAN controller and cleanup resources."""
        self.__running = False
        if self.__thread and self.__thread.is_alive():
            self.__thread.join(timeout=2)

    def __listen(self) -> None:
        """Internal message listening loop for feedback messages."""
        while self.__running:
            message = self.can_bus.recv(0.5)
            if message is not None and message.arbitration_id in self.__listeners:
                for listener in self.__listeners[message.arbitration_id]:
                    listener(message)