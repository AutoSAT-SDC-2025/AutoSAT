import struct
import can
from src.can_interface.can_controller_interface import ICanController, init_can_message
from src.car_variables import KartControlCanIDs, CAN_MESSAGE_SENDING_SPEED, KartGearBox

class KartCANController(ICanController):

    can_bus: can.Bus
    __listeners: dict[int, list[callable]]

    def __init__(self, can_bus: can.Bus) -> None:
        self.can_bus = can_bus
        self.__listeners = {}

        self.__kart_gearbox = KartGearBox.neutral

        self.__throttle_message = init_can_message(KartControlCanIDs.throttle.value)
        self.__throttle_task = self.can_bus.send_periodic(self.__throttle_message, CAN_MESSAGE_SENDING_SPEED)

        self.__steering_message = init_can_message(KartControlCanIDs.steering.value)
        self.__steering_task = self.can_bus.send_periodic(self.__steering_message, CAN_MESSAGE_SENDING_SPEED)

        self.__breaking_message = init_can_message(KartControlCanIDs.breaking.value)
        self.__breaking_task = self.can_bus.send_periodic(self.__breaking_message, CAN_MESSAGE_SENDING_SPEED)

    def add_listener(self, message_id: int, listener: callable) -> None:
        if message_id not in self.__listeners:
            self.__listeners[message_id] = []
        self.__listeners[message_id].append(listener)

    async def set_steering(self, steering_angle: float) -> None:
        little_endian_bytes = struct.pack('f', steering_angle)
        self.__steering_message.data = list(bytearray(little_endian_bytes)) + [0, 0, 195, 0]
        self.__steering_task.modify_data(self.__steering_message)

    async def set_kart_gearbox(self, kart_gearbox: KartGearBox) -> None:
        self.__kart_gearbox = kart_gearbox

    async def set_throttle(self, throttle_value: float) -> None:
        self.__throttle_message.data = [int(throttle_value), 0, self.__kart_gearbox.value, 0, 0, 0, 0, 0]
        self.__throttle_task.modify_data(self.__throttle_message)

    async def set_break(self, break_value: int) -> None:
        self.__breaking_message.data[0] = break_value
        self.__breaking_task.modify_data(self.__breaking_message)

    def __listen(self) -> None:
        while True:
            message = self.can_bus.recv(0.5)
            if message is not None and message.arbitration_id in self.__listeners:
                for listener in self.__listeners[message.arbitration_id]:
                    listener(message)