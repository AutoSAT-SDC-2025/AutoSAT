from abc import ABC, abstractmethod
import struct
from enum import IntEnum

import can
from src.car_variables import KartControlCanIDs, KartFeedbackCanIDs, KartGearBox, HunterControlCanIDs, HunterFeedbackCanIDs, HunterControlMode ,CAN_MESSAGE_SENDING_SPEED, CarType

def init_can_message(message_id: int) -> can.Message:
    return can.Message(arbitration_id=message_id, data=[0, 0, 0, 0, 0, 0, 0, 0], is_extended_id=False)

class ICanController(ABC):

    @abstractmethod
    def send_movement(self, throttle_value: float, kart_gear: KartGearBox, steering_angle: float) -> None:
        pass

    @abstractmethod
    def send_control(self, break_value: int, parking_value: bool, control_mode: HunterControlMode) -> None:
        pass

    @abstractmethod
    def monitor_bus(self) -> None:
        pass

class CarCanController(ICanController):

    def __init__(self, bus: can.Bus, car_type: CarType) -> None:
        self._car_type = car_type
        if self._car_type is CarType.kart:

            self.kart_gear = 1
            self.can_bus = bus

            self.throttle_message = init_can_message(KartControlCanIDs.throttle.value & 0x1FFFFFFF)
            self.throttle_task = self.can_bus.send_periodic(self.throttle_message, CAN_MESSAGE_SENDING_SPEED)

            self.steering_message = init_can_message(KartControlCanIDs.steering.value & 0x1FFFFFFF)
            self.steering_task = self.can_bus.send_periodic(self.steering_message, CAN_MESSAGE_SENDING_SPEED)

            self.breaking_message = init_can_message(KartControlCanIDs.breaking.value & 0x1FFFFFFF)
            self.breaking_task = self.can_bus.send_periodic(self.breaking_message, CAN_MESSAGE_SENDING_SPEED)

        elif self._car_type is CarType.hunter:

            self.can_bus = bus

            self.movement_control_message = init_can_message(HunterControlCanIDs.movement_control.value)
            self.movement_control_task = self.can_bus.send_periodic(self.movement_control_message,
                                                                    CAN_MESSAGE_SENDING_SPEED)

            self.parking_control_message = init_can_message(HunterControlCanIDs.parking_control.value)
            self.parking_control_task = self.can_bus.send_periodic(self.parking_control_message,
                                                                   CAN_MESSAGE_SENDING_SPEED)

            self.control_mode_message = init_can_message(HunterControlCanIDs.control_mode.value)
            self.control_mode_task = self.can_bus.send_periodic(self.control_mode_message, CAN_MESSAGE_SENDING_SPEED)

    async def send_movement(self, throttle_value: float, kart_gear: KartGearBox, steering_angle: float) -> None:

        if self._car_type is CarType.kart:

            self.throttle_message.data = [int(throttle_value), 0, kart_gear.value, 0, 0, 0, 0, 0]
            self.throttle_task.modify_data(self.throttle_message)

            little_endian_bytes = struct.pack('f', steering_angle)
            self.steering_message.data = list(bytearray(little_endian_bytes)) + [0, 0, 195, 0]
            self.steering_task.modify_data(self.steering_message)

        elif self._car_type is CarType.hunter:

            speed_bytes = struct.pack('>h', int(throttle_value))
            steering_angle_bytes = struct.pack('>h', int(steering_angle))
            self.movement_control_message.data = list(bytearray(speed_bytes)) + [0, 0, 0, 0] + list(
                bytearray(steering_angle_bytes))
            self.movement_control_task.modify_data(self.movement_control_message)

    async def send_control(self, break_value: int, parking_value: bool, control_mode: HunterControlMode) -> None:
        if self._car_type is CarType.kart:

            self.breaking_message.data[0] = break_value
            self.breaking_task.modify_data(self.breaking_message)

        elif self._car_type is CarType.hunter:

            self.parking_control_message.data = list(bytearray(struct.pack('?', parking_value)))
            self.parking_control_task.modify_data(self.parking_control_message)

            self.control_mode_message.data = list(bytearray(struct.pack('?', control_mode.value)))
            self.control_mode_task.modify_data(self.control_mode_message)

    async def monitor_bus(self) -> can.Message | None:
        if self._car_type is CarType.kart:
            self.can_bus.set_filters(
                [{"can_id": KartFeedbackCanIDs.internal_throttle, "can_mask": 0xFFF, "extended": False},
                 {"can_id": KartFeedbackCanIDs.steering_ecu, "can_mask": 0xFFF, "extended": False},
                 {"can_id": KartFeedbackCanIDs.steering_sensor, "can_mask": 0xFFF, "extended": False},
                 {"can_id": KartFeedbackCanIDs.breaking_sensor, "can_mask": 0xFFF, "extended": False}]
            )
            message = self.can_bus.recv(1)
            return message

        elif self._car_type is CarType.hunter:
            self.can_bus.set_filters(
                [{"can_id": HunterFeedbackCanIDs.status_feedback, "can_mask": 0xFFF, "extended": False},
                 {"can_id": HunterFeedbackCanIDs.movement_feedback, "can_mask": 0xFFF, "extended": False}]
            )
            message = self.can_bus.recv(1)
            return message