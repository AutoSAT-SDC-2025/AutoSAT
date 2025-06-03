from abc import ABC, abstractmethod
import can

def init_can_message(message_id: int) -> can.Message:
    return can.Message(arbitration_id=message_id, data=[0, 0, 0, 0, 0, 0, 0, 0], is_extended_id=False)

class ICanController(ABC):

    @abstractmethod
    def add_listener(self, message_id: int, listener: callable) -> None:
        pass

    @abstractmethod
    def set_throttle(self, throttle_value: float) -> None:
        pass

    @abstractmethod
    def set_steering(self, steering_angle: float) -> None:
        pass

    @abstractmethod
    def set_break(self, break_value: int) -> None:
        pass

    @abstractmethod
    def set_kart_gearbox(self, kart_gearbox) -> None:
        pass

    @abstractmethod
    def start(self) -> None:
        pass

    @abstractmethod
    def stop(self) -> None:
        pass