import logging

from ..IControlMode import IControlMode
from ...can_interface.bus_connection import connect_to_can_interface, disconnect_from_can_interface
from ...can_interface.can_factory import CanControllerFactory
from ...car_variables import CarType, HunterControlMode, KartGearBox, HunterFeedbackCanIDs, KartFeedbackCanIDs
from ...gamepad import Gamepad
from ...gamepad.controller_mapping import ControllerMapping
from ...misc import calculate_steering, calculate_throttle

class ManualMode(IControlMode):

    def __init__(self, car_type: CarType):
        self.can_bus = connect_to_can_interface(0)
        self.car_type = car_type
        self.can_controller = CanControllerFactory.create_can_controller(self.can_bus, self.car_type)

        if Gamepad.available():
            print("Connected to Gamepad")
            self.gamepad = Gamepad.XboxONE()
            self.gamepad.startBackgroundUpdates()
        else:
            print('Controller not connected :(')
            return

    async def car_input(self):
        if self.gamepad.beenPressed(ControllerMapping.buttonExit):
            print("exiting....")
            await self.stop()
            return None, None, None
        steering = calculate_steering(self.gamepad.axis(ControllerMapping.L_joystickX), self.car_type)
        throttle = calculate_throttle(self.gamepad.axis(ControllerMapping.R_joystickY), self.car_type)
        park = dead_man_switch(self.gamepad)
        return steering, throttle, park

    async def start(self) -> None:
        try:
            if self.car_type == CarType.hunter:
                await self.can_controller.set_control_mode(HunterControlMode.command_mode)
            elif self.car_type == CarType.kart:
                await self.can_controller.set_kart_gearbox(KartGearBox.forward)

            while self.gamepad.isConnected():
                steering, throttle, park = await self.car_input()
                if steering is not None and throttle is not None and park is not None:
                    if self.car_type == CarType.hunter:
                        self.can_controller.add_listener(HunterFeedbackCanIDs.movement_feedback, print_can_messages)
                        self.can_controller.add_listener(HunterFeedbackCanIDs.status_feedback, print_can_messages)
                        await self.can_controller.set_throttle(throttle)
                        await self.can_controller.set_steering(steering)
                        await self.can_controller.set_parking_mode(park)
                    elif self.car_type == CarType.kart:
                        self.can_controller.add_listener(KartFeedbackCanIDs.steering_ecu, print_can_messages)
                        self.can_controller.add_listener(KartFeedbackCanIDs.steering_sensor, print_can_messages)
                        self.can_controller.add_listener(KartFeedbackCanIDs.breaking_sensor, print_can_messages)
                        self.can_controller.add_listener(KartFeedbackCanIDs.internal_throttle, print_can_messages)
                        await self.can_controller.set_throttle(throttle)
                        await self.can_controller.set_steering(steering)
                        await self.can_controller.set_break(controller_break_value(self.gamepad))
                    print(f"{steering} \t {throttle}")
                else:
                    break
        finally:
            await self.stop()


    async def stop(self) -> None:
        self.gamepad.disconnect()
        if self.car_type == CarType.hunter:
            await self.can_controller.set_control_mode(HunterControlMode.idle_mode)
        else:
            await self.can_controller.set_kart_gearbox(KartGearBox.neutral)
            await self.can_controller.set_break(100)
        disconnect_from_can_interface(self.can_bus)
        print("Stopping manual mode")

def dead_man_switch(gamepad: Gamepad) -> bool:
    if gamepad.axis(ControllerMapping.park) == 1:
        park = False
        logging.debug("Dead Man Switch pressed")
    else:
        park = True
        logging.debug("Dead Man Switch not pressed")
    return park

def controller_break_value(gamepad: Gamepad) -> int:
    return max(0,-round((gamepad.axis(ControllerMapping.park)**3)*100))

def print_can_messages(message) -> None:
    print(f"Throttle: {message}\t")