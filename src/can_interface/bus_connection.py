"""
CAN bus connection module for automotive communication.

This module provides functions to establish and manage connections to CAN (Controller Area Network)
interfaces, supporting both physical and virtual CAN buses for vehicle communication.
"""

import logging
import can
import os


def connect_to_can_interface(bus_id: int) -> can.ThreadSafeBus:
    """
    Establish connection to a CAN interface.
    
    Attempts to connect to a physical CAN interface first. If the physical
    interface is not available, falls back to creating a virtual CAN interface
    for testing and development purposes.
    
    Args:
        bus_id: The ID number of the CAN bus (e.g., 0 for can0, 1 for can1)
        
    Returns:
        A ThreadSafeBus object connected to either physical or virtual CAN interface
        
    Note:
        - Physical interface uses socketcan with specified bitrate
        - Virtual interface (vcan0) is created if physical interface fails
        - Requires sudo privileges for interface configuration
    """
    bus_channel = "can" + str(bus_id)
    bitrate = 500000
    
    # Check if physical CAN interface exists
    if os.system(f"ip link show {bus_channel}") == 0:
        # Configure physical CAN interface
        os.system(f"sudo ip link set {bus_channel} type can bitrate {bitrate}")
        os.system(f"sudo ip link set {bus_channel} up")
        logging.debug(f"Initialized physical CAN interface: {bus_channel}")
        return can.ThreadSafeBus(interface='socketcan', channel=bus_channel, bitrate=bitrate)
    
    # Fallback to virtual CAN interface
    logging.error(
        f'Interface {bus_channel} not available. Error when initialising.\nCreating a virtual interface instead.')
    os.system("sudo ip link add dev vcan0 type vcan")
    os.system("sudo ip link set dev vcan0 up")
    logging.debug("Initialized virtual CAN interface: vcan0")
    return can.ThreadSafeBus(interface='virtual', channel='vcan0', bitrate=bitrate, receive_own_messages=True)


def disconnect_from_can_interface(can_bus: can.ThreadSafeBus) -> None:
    """
    Safely disconnect from the CAN interface.
    
    Properly shuts down the CAN bus connection to free system resources
    and ensure clean termination of the communication channel.
    
    Args:
        can_bus: The ThreadSafeBus object to disconnect from
        
    Note:
        This function should always be called before program termination
        to prevent resource leaks and ensure proper cleanup.
    """
    logging.debug("Shutting down CAN interface.")
    can_bus.shutdown()
