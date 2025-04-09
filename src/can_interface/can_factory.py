from abc import ABC, abstractmethod
from src.car_variables import CarType
from src.can_interface.can_controller_interface import ICanController
from src.can_interface.kart_can_controller import KartCANController
from src.can_interface.hunter_can_controller import HunterCanController

class CanControllerCreator(ABC):
    @abstractmethod
    def factory_method(self, can_bus):
        pass

class KartCANControllerCreator(CanControllerCreator):
    def factory_method(self, can_bus) -> KartCANController:
        return KartCANController(can_bus)

class HunterCANControllerCreator(CanControllerCreator):
    def factory_method(self, can_bus) -> HunterCanController:
        return HunterCanController(can_bus)

def create_can_controller(can_creator: CanControllerCreator, can_bus):
    return can_creator.factory_method(can_bus)

def select_can_controller_creator(car_type: CarType):
    if car_type is CarType.hunter:
        return HunterCANControllerCreator()
    elif car_type is CarType.kart:
        return KartCANControllerCreator()
    else:
        raise ValueError(f"Unknown car type: {car_type}")
