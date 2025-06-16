"""
CAN controller interface module for autonomous vehicle communication.

This module defines the abstract interface for CAN (Controller Area Network) controllers
and provides utility functions for CAN message initialization. It establishes the contract
that all CAN controller implementations must follow for vehicle control operations in the
AutoSAT autonomous vehicle system.

The interface supports multiple vehicle types including Hunter and Kart configurations,
each with their own specific CAN message protocols and control characteristics.
"""

from abc import ABC, abstractmethod
import can


def init_can_message(message_id: int) -> can.Message:
    """
    Initialize a CAN message with default values for vehicle communication.
    
    Creates a standard CAN message with the specified arbitration ID and
    initializes all data bytes to zero. Uses standard (11-bit) identifier format
    as required by the AutoSAT vehicle communication protocol.
    
    Args:
        message_id: The CAN arbitration ID for the message. Should correspond to
                   predefined message IDs from HunterControlCanIDs or KartControlCanIDs
        
    Returns:
        A CAN Message object with 8 zero-initialized data bytes ready for data population
        
    Example:
        >>> msg = init_can_message(0x123)
        >>> msg.data = [throttle_value, 0, 0, 0, 0, 0, 0, 0]
    """
    return can.Message(arbitration_id=message_id, data=[0, 0, 0, 0, 0, 0, 0, 0], is_extended_id=False)


class ICanController(ABC):
    """
    Abstract interface for CAN controller implementations in the AutoSAT system.
    
    This abstract base class defines the required methods that all CAN controller
    implementations must provide for vehicle control operations. It supports both
    Hunter and Kart vehicle types, each with different control characteristics:
    
    - Hunter: Uses combined steering/throttle commands, supports parking brake,
              has multiple control modes (idle, command, remote)
    - Kart: Uses separate steering and throttle commands, has gearbox control,
            simpler brake system
    
    The interface handles real-time vehicle control through CAN bus communication,
    supporting operations like throttle control, steering adjustment, braking,
    and vehicle-specific features like gearbox management.
    
    Implementations must handle:
    - Message encoding/decoding for specific vehicle protocols
    - Real-time command transmission with appropriate timing
    - Listener registration for feedback messages
    - Proper resource management and cleanup
    """

    @abstractmethod
    def add_listener(self, message_id: int, listener: callable) -> None:
        """
        Register a callback for incoming CAN messages with specific arbitration ID.
        
        Allows registration of listener functions to handle feedback messages from
        vehicle systems. Common use cases include monitoring vehicle status,
        receiving sensor data, and tracking command acknowledgments.
        
        Args:
            message_id: The CAN message arbitration ID to monitor. Should match
                       feedback message IDs from HunterFeedbackCanIDs or KartFeedbackCanIDs
            listener: Callback function that will be invoked when messages with the
                     specified ID are received. Function signature should accept
                     a can.Message parameter
                     
        Example:
            >>> def status_handler(msg):
            ...     print(f"Status: {msg.data}")
            >>> controller.add_listener(0x456, status_handler)
        """
        pass

    @abstractmethod
    def set_throttle(self, throttle_value: float) -> None:
        """
        Control vehicle acceleration/deceleration through throttle commands.
        
        Sends throttle control messages via CAN bus to manage vehicle speed.
        
        Args:
            throttle_value: Desired throttle position. Range and units depend on
                           vehicle type.
        """
        pass

    @abstractmethod
    def set_steering(self, steering_angle: float) -> None:
        """
        Control vehicle steering angle for directional navigation.
        
        Sends steering commands via CAN messages to the vehicle's steering system.

        Args:
            steering_angle: Desired steering angle. Units and range depend on
                           vehicle type.
        """
        pass

    @abstractmethod
    def set_break(self, break_value: int) -> None:
        """
        Engage vehicle braking system for deceleration and stopping.
        
        Controls the vehicle's braking system through CAN messages.
        
        Args:
            break_value: boolean.
        """
        pass

    @abstractmethod
    def set_kart_gearbox(self, kart_gearbox) -> None:
        """
        Control gearbox configuration for kart-type vehicles.
        
        Manages transmission settings specific to kart vehicles through CAN
        commands. Supports different gear states like forward, neutral, and
        reverse operations.
        
        Args:
            kart_gearbox: Gearbox configuration, typically a KartGearBox enum value:
                         - KartGearBox.forward: Forward gear engagement
                         - KartGearBox.neutral: Neutral position
                         - KartGearBox.reverse: Reverse gear
                         
        Note:
            This method is kart-specific.
        """
        pass

    @abstractmethod
    def start(self) -> None:
        """
        Initialize and start the CAN controller communication system.
        
        Establishes CAN bus connection, initializes message handlers, and begins
        real-time communication with the vehicle. This method must be called
        before any vehicle control operations can be performed.

        """
        pass

    @abstractmethod
    def stop(self) -> None:
        """
        Safely shutdown the CAN controller and clean up resources.
        
        Performs orderly shutdown of CAN communication, stops all control
        threads, and releases system resources. Should be called before
        program termination to ensure proper cleanup and avoid resource leaks.

        """
        pass