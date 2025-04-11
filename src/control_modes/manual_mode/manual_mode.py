import logging
import time

from ..IControlMode import IControlMode
from ...can_interface.bus_connection import connect_to_can_interface, disconnect_from_can_interface
from ...can_interface.can_factory import select_can_controller_creator, create_can_controller
from ...car_variables import CarType, HunterControlMode, KartGearBox
from ...gamepad import Gamepad
from ...gamepad.controller_mapping import ControllerMapping
from ...misc import calculate_steering, calculate_throttle, controller_break_value, dead_man_switch, setup_listeners


class ManualMode(IControlMode):

    def __init__(self, car_type: CarType):
        self.__running = True
        self.can_bus = connect_to_can_interface(0)
        self.car_type = car_type
        can_creator = select_can_controller_creator(self.car_type)
        self.can_controller = create_can_controller(can_creator, self.can_bus)
        setup_listeners(self.can_controller, self.car_type)
        
        self.steering = 0.0
        self.throttle = 0.0
        self.park = True

        if Gamepad.available():
            logging.info("Connected to Gamepad")
            self.gamepad = Gamepad.XboxOne()
            self.gamepad.startBackgroundUpdates()
            
            self.gamepad.addAxisMovedHandler(ControllerMapping.L_joystickX, self.handle_steering)
            self.gamepad.addAxisMovedHandler(ControllerMapping.R_joystickY, self.handle_throttle)
            self.gamepad.addAxisMovedHandler(ControllerMapping.park, self.handle_park)
            
            self.gamepad.addButtonPressedHandler(ControllerMapping.buttonExit, self.stop_manual_mode)


            self.gamepad.rumble(strong_magnitude=30000, weak_magnitude=30000, duration_ms=1000)
        else:
            logging.info('Controller not connected :(')
            self.gamepad = None

    def handle_steering(self, value):
        """Callback for steering axis changes"""
        self.steering = calculate_steering(value, self.car_type)

    def handle_throttle(self, value):
        """Callback for throttle axis changes"""
        self.throttle = calculate_throttle(value, self.car_type)

    def handle_park(self, value):
        """Callback for parking trigger"""
        trigger_threshold = 0.8
        self.park = True if value < trigger_threshold else False

    def start(self) -> None:
        """Start manual mode."""
        logging.info("Starting manual mode...")
        try:
            self.can_controller.start()
            while self.gamepad and self.gamepad.isConnected() and self.__running:
                if self.car_type == CarType.hunter:
                    self.can_controller.set_control_mode(HunterControlMode.command_mode)
                    self.can_controller.set_steering_and_throttle(self.steering, self.throttle)
                    self.can_controller.set_parking_mode(self.park)
                elif self.car_type == CarType.kart:
                    self.can_controller.set_kart_gearbox(KartGearBox.forward)
                    self.can_controller.set_throttle(self.throttle)
                    self.can_controller.set_steering(self.steering)
                    self.can_controller.set_break(controller_break_value(self.gamepad))
                time.sleep(0.01)
        except Exception as e:
            logging.error(f"Error in manual mode: {e}")
        finally:
            self.stop()

    def stop(self) -> None:
        """Stop manual mode."""
        if self.gamepad and self.gamepad.isConnected():
            self.gamepad.disconnect()
        if self.car_type == CarType.hunter:
            self.can_controller.set_control_mode(HunterControlMode.idle_mode)
        else:
            self.can_controller.set_kart_gearbox(KartGearBox.neutral)
            self.can_controller.set_break(100)
        disconnect_from_can_interface(self.can_bus)
        logging.info("Exiting... \nStopping manual mode")

    def stop_manual_mode(self):
        logging.info("Exit button pressed. Exiting manual mode.")
        self.__running = False