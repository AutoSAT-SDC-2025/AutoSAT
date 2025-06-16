"""
Vehicle configuration constants and enumerations for AutoSAT.

Defines CAN message IDs, control modes, vehicle types, and system parameters
for Hunter and Kart autonomous vehicles. Provides centralized configuration
for communication protocols and hardware specifications.
"""

from enum import IntEnum

# CAN message transmission frequency (seconds between messages)
CAN_MESSAGE_SENDING_SPEED = 0.08  # 0.04 or 0.06 alternative rates

class HunterControlMode(IntEnum):
    """
    Hunter vehicle control mode states.
    
    Defines operational modes for Hunter vehicle command interface.
    """
    idle_mode = 0      # Vehicle in standby, ignoring movement commands
    command_mode = 1   # Vehicle accepting external control commands

class HunterControlCanIDs(IntEnum):
    """
    Hunter vehicle CAN message IDs for control commands.
    
    Outbound message identifiers for sending commands to Hunter vehicle.
    """
    movement_control = 0x111   # Combined speed and steering commands
    control_mode = 0x421       # Control mode switching (idle/command)
    parking_control = 0x131    # Parking brake engagement control

class HunterFeedbackCanIDs(IntEnum):
    """
    Hunter vehicle CAN message IDs for feedback data.
    
    Inbound message identifiers for receiving status from Hunter vehicle.
    """
    movement_feedback = 0x221  # Current speed and steering position
    status_feedback = 0x211    # Vehicle status, control mode, brake state

class KartControlCanIDs(IntEnum):
    """
    Kart vehicle CAN message IDs for control commands.
    
    Outbound message identifiers for sending commands to Kart vehicle.
    Kart uses separate control channels for each system.
    """
    breaking = 0x110   # Brake actuator position commands
    steering = 0x220   # Steering angle commands
    throttle = 0x330   # Throttle and gear control commands

class KartFeedbackCanIDs(IntEnum):
    """
    Kart vehicle CAN message IDs for feedback data.
    
    Inbound message identifiers for receiving sensor data from Kart vehicle.
    """
    steering_sensor = 0x1E5     # Raw steering position sensor
    breaking_sensor = 0x710     # Brake actuator position and status
    speed_sensor = 0x440        # Vehicle speed from the speed sensor
    internal_throttle = 0x730   # Throttle position and drivetrain status

class KartGearBox(IntEnum):
    """
    Kart transmission gear states.
    
    Available gear positions for Kart vehicle drivetrain control.
    """
    neutral = 0   # No power transmission (park/neutral)
    forward = 1   # Forward gear engagement
    backward = 2  # Reverse gear engagement

class CarType(IntEnum):
    """
    Vehicle type identification for system configuration.
    
    Determines which control protocol and CAN IDs to use.
    """
    kart = 0    # Go-kart style vehicle with separate control systems
    hunter = 1  # Hunter AGV with integrated control systems

class ControlMode(IntEnum):
    """
    High-level control mode selection for vehicle operation.
    
    Determines source of vehicle commands and operational behavior.
    """
    automatic = 0  # Autonomous operation via computer vision/AI
    manual = 1     # Direct operator control via gamepad/interface

class CameraResolution(IntEnum):
    """
    Camera system resolution and calibration parameters.
    
    Standard resolution settings for computer vision processing
    and camera calibration constants.
    """
    WIDTH = 848        # Camera frame width in pixels
    HEIGHT = 480       # Camera frame height in pixels
    FOCAL_LENGTH = 540 # Camera focal length for depth calculations

class LineDetectionDims(IntEnum):
    """
    Image processing dimensions for line detection algorithms.
    
    Square processing region for lane detection and path following
    computer vision operations.
    """
    WIDTH = 720   # Processing region width in pixels
    HEIGHT = 720  # Processing region height in pixels