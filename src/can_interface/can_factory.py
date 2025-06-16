"""
CAN controller factory module for vehicle-specific controller creation.

This module implements the Factory Method pattern to create appropriate CAN controllers
based on vehicle type (Hunter or Kart), providing a clean interface for controller
instantiation without tight coupling to specific implementations.
"""

from abc import ABC, abstractmethod
from src.car_variables import CarType
from src.can_interface.can_controller_interface import ICanController
from src.can_interface.kart_can_controller import KartCANController
from src.can_interface.hunter_can_controller import HunterCanController


class CanControllerCreator(ABC):
    """
    Abstract factory for CAN controller creation.
    
    Defines the interface for concrete factory classes that create
    vehicle-specific CAN controllers.
    """
    
    @abstractmethod
    def factory_method(self, can_bus):
        """
        Create a CAN controller instance.
        
        Args:
            can_bus: CAN bus interface for controller communication
            
        Returns:
            Vehicle-specific CAN controller implementation
        """
        pass


class KartCANControllerCreator(CanControllerCreator):
    """Factory for creating Kart CAN controllers."""
    
    def factory_method(self, can_bus) -> KartCANController:
        """Create a KartCANController instance."""
        return KartCANController(can_bus)


class HunterCANControllerCreator(CanControllerCreator):
    """Factory for creating Hunter CAN controllers."""
    
    def factory_method(self, can_bus) -> HunterCanController:
        """Create a HunterCanController instance."""
        return HunterCanController(can_bus)


def create_can_controller(can_creator: CanControllerCreator, can_bus):
    """
    Create CAN controller using provided factory.
    
    Args:
        can_creator: Factory instance for controller creation
        can_bus: CAN bus interface for controller communication
        
    Returns:
        Created CAN controller instance
    """
    return can_creator.factory_method(can_bus)


def select_can_controller_creator(car_type: CarType):
    """
    Select appropriate factory based on vehicle type.
    
    Args:
        car_type: Type of vehicle (CarType.hunter or CarType.kart)
        
    Returns:
        Appropriate factory creator instance
        
    Raises:
        ValueError: If car_type is not supported
    """
    if car_type is CarType.hunter:
        return HunterCANControllerCreator()
    elif car_type is CarType.kart:
        return KartCANControllerCreator()
    else:
        raise ValueError(f"Unknown car type: {car_type}")
