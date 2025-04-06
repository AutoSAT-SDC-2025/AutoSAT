from src.car_variables import CarType
from src.can_interface.can_controller_interface import ICanController
from src.can_interface.kart_can_controller import KartCANController
from src.can_interface.hunter_can_controller import HunterCanController

class CanControllerFactory:
    @staticmethod
    def create_can_controller(can_bus, car_type: CarType) -> ICanController:
        if car_type is CarType.kart:
            return KartCANController(can_bus)
        elif car_type is CarType.hunter:
            return HunterCanController(can_bus)
        else:
            raise ValueError(f'Unknown car type: {car_type}')