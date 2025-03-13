import asyncio
from src.gamepad import Gamepad
from src.misc import calc_axis_angle, calculate_hunter_throttle, calculate_hunter_steering
from src.gamepad.controller_mapping import ControllerMapping
from src.can_interface.can_controller import CarCanController, CarType, HunterControlMode
from src.can_interface.bus_connection import connect_to_can_interface
from car_variables import KartGearBox

async def main() -> None:

    can_bus = connect_to_can_interface(0)
    can_controller = CarCanController(can_bus, CarType.hunter)

    if Gamepad.available():
        print("Connected to Gamepad")
        gamepad = Gamepad.Xbox360()
        gamepad.startBackgroundUpdates()
    else:
        print('Controller not connected :(')
        return

    try:
        while gamepad.isConnected():
            if gamepad.beenPressed(ControllerMapping.buttonExit):
                print("exiting....")
                break
            steering = calculate_hunter_steering(gamepad.axis(ControllerMapping.L_joystickX), CarType.hunter)
            throttle = calculate_hunter_throttle(gamepad.axis(ControllerMapping.R_joystickY), CarType.hunter)

            await can_controller.send_control(0, False, HunterControlMode.command_mode)
            await can_controller.send_movement(throttle, KartGearBox.neutral, steering)
            message = await can_controller.monitor_bus()

            print(f"{steering} \t {throttle} \t {message}")
            # print(f"{steering} \t {throttle}")
    finally:
        gamepad.disconnect()

if __name__ == "__main__":
    asyncio.run(main())