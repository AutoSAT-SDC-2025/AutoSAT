import logging
from ..IControlMode import IControlMode
from ...can_interface.bus_connection import connect_to_can_interface, disconnect_from_can_interface
from ...can_interface.can_factory import select_can_controller_creator, create_can_controller
from ...car_variables import CarType, HunterControlMode, KartGearBox
from ...gamepad import Gamepad
from ...gamepad.controller_mapping import ControllerMapping
from ...misc import calculate_steering, calculate_throttle, controller_break_value, dead_man_switch, setup_listeners


class ManualMode(IControlMode):

    def __init__(self, car_type: CarType):
        self.can_bus = connect_to_can_interface(0)
        self.car_type = car_type
        can_creator = select_can_controller_creator(self.car_type)
        self.can_controller = create_can_controller(can_creator, self.can_bus)
        setup_listeners(self.can_controller, self.car_type)

        if Gamepad.available():
            logging.info("Connected to Gamepad")
            self.gamepad = Gamepad.XboxOne()
            self.gamepad.startBackgroundUpdates()
            self.gamepad.rumble(strong_magnitude=30000, weak_magnitude=30000, duration_ms=1000)
        else:
            logging.info('Controller not connected :(')
            self.gamepad = None

    def car_input(self):
        """Get inputs from the gamepad."""
        steering = calculate_steering(self.gamepad.axis(ControllerMapping.L_joystickX), self.car_type)
        throttle = calculate_throttle(self.gamepad.axis(ControllerMapping.R_joystickY), self.car_type)
        park = dead_man_switch(self.gamepad)
        return steering, throttle, park

    def start(self) -> None:
        """Start manual mode."""
        logging.info("Starting manual mode...")
        try:
            self.can_controller.start()
            while self.gamepad and self.gamepad.isConnected():
                if self.gamepad.beenPressed(ControllerMapping.buttonExit):
                    logging.info("Exit button pressed. Exiting manual mode.")
                    break

                if self.car_type == CarType.hunter:
                    self.can_controller.set_control_mode(HunterControlMode.command_mode)
                elif self.car_type == CarType.kart:
                    self.can_controller.set_kart_gearbox(KartGearBox.forward)

                steering, throttle, park = self.car_input()

                if steering is None or throttle is None or park is None:
                    logging.warning("Invalid inputs received. Exiting loop.")
                    break

                if self.car_type == CarType.hunter:
                    self.can_controller.set_steering_and_throttle(steering, throttle)
                    self.can_controller.set_parking_mode(park)
                elif self.car_type == CarType.kart:
                    self.can_controller.set_throttle(throttle)
                    self.can_controller.set_steering(steering)
                    self.can_controller.set_break(controller_break_value(self.gamepad))

                # logging.info(f"Steering: {steering}, Throttle: {throttle}, Park: {park}")

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