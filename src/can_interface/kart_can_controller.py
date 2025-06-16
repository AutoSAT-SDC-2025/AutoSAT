"""
Kart vehicle CAN controller implementation.

Provides CAN bus communication interface for Kart vehicles with separate
steering, throttle, brake, and gearbox control commands.
"""

import logging
import struct
import threading

import can
from src.can_interface.can_controller_interface import ICanController, init_can_message
from src.car_variables import KartControlCanIDs, CAN_MESSAGE_SENDING_SPEED, KartGearBox

class KartCANController(ICanController):
    """
    CAN controller implementation for Kart vehicles.
    
    Manages separate periodic messages for steering, throttle, brake, and gearbox.
    Tracks current state to avoid redundant CAN traffic and provides listener
    interface for feedback messages.
    """
    
    can_bus: can.Bus
    __listeners: dict[int, list[callable]]
    __thread: threading.Thread
    __running: bool

    def __init__(self, bus: can.Bus) -> None:
        """
        Initialize Kart CAN controller with periodic message tasks.
        
        Args:
            bus: CAN bus interface for communication
        """
        self.can_bus = bus
        self.__listeners = {}

        self.__kart_gearbox = KartGearBox.neutral
        self.__steering_angle = 0.0
        self.__throttle_value = 0.0
        self.__brake_value = 0

        self.__throttle_message = init_can_message(KartControlCanIDs.throttle.value)
        self.__throttle_task = self.can_bus.send_periodic(self.__throttle_message, CAN_MESSAGE_SENDING_SPEED)

        self.__steering_message = init_can_message(KartControlCanIDs.steering.value)
        self.__steering_task = self.can_bus.send_periodic(self.__steering_message, CAN_MESSAGE_SENDING_SPEED)

        self.__breaking_message = init_can_message(KartControlCanIDs.breaking.value)
        self.__breaking_task = self.can_bus.send_periodic(self.__breaking_message, CAN_MESSAGE_SENDING_SPEED)
        self.__running = False

    def add_listener(self, message_id: int, listener: callable) -> None:
        """
        Register listener for specific feedback message ID.
        
        Args:
            message_id: Kart feedback message ID to monitor
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
        logging.debug("created task for __listen")

    def set_steering(self, steering_angle: float) -> None:
        """
        Set kart steering angle.
        
        Args:
            steering_angle: Steering angle (±1.25)
        """
        if steering_angle == self.__steering_angle:
            return
        self.__steering_angle = steering_angle
        little_endian_bytes = struct.pack('f', steering_angle)
        self.__steering_message.data = list(bytearray(little_endian_bytes)) + [0, 0, 195, 0]
        self.__steering_task.modify_data(self.__steering_message)

    def set_kart_gearbox(self, kart_gearbox: KartGearBox) -> None:
        """
        Set kart gearbox state.
        
        Args:
            kart_gearbox: Gear selection (neutral, forward, reverse)
        """
        if kart_gearbox == self.__kart_gearbox:
            return
        self.__kart_gearbox = kart_gearbox
        self.set_throttle(self.__throttle_value)

    def set_throttle(self, throttle_value: float) -> None:
        """
        Set kart throttle position.
        
        Args:
            throttle_value: Throttle value (0-100)
        """
        if throttle_value == self.__throttle_value:
            return
        self.__throttle_value = throttle_value
        self.__throttle_message.data = [int(self.__throttle_value), 0, self.__kart_gearbox.value, 0, 0, 0, 0, 0]
        self.__throttle_task.modify_data(self.__throttle_message)

    def set_break(self, break_value: int) -> None:
        """
        Set kart brake intensity.
        
        Args:
            break_value: Brake value (0-100)
        """
        if break_value == self.__brake_value:
            return
        self.__brake_value = break_value
        self.__breaking_message.data[0] = break_value
        self.__breaking_task.modify_data(self.__breaking_message)

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

    def get_steering_angle(self) -> float:
        """Get current steering angle."""
        return self.__steering_angle

    def get_throttle_value(self) -> float:
        """Get current throttle value."""
        return self.__throttle_value

    def get_brake_value(self) -> int:
        """Get current brake value."""
        return self.__brake_value

    def get_gearbox_state(self) -> KartGearBox:
        """Get current gearbox state."""
        return self.__kart_gearbox