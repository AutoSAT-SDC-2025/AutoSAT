"""
Abstract interface for vehicle control modes.

Defines the contract that all control mode implementations must follow
for starting and stopping vehicle control operations.
"""

from abc import ABC, abstractmethod

class IControlMode(ABC):
    """
    Abstract base class for vehicle control modes.
    
    Provides interface for different control strategies like manual gamepad control,
    autonomous driving, line following, and obstacle avoidance modes.
    """

    @abstractmethod
    def start(self) -> None:
        """
        Start the control mode operation.
        
        Initializes the control system and begins vehicle control operations.
        Implementation should handle setup, main control loop, and error handling.
        """
        pass

    @abstractmethod
    def stop(self) -> None:
        """
        Stop the control mode and cleanup resources.
        
        Safely terminates control operations, sets vehicle to safe state,
        and releases system resources like CAN connections and sensors.
        """
        pass