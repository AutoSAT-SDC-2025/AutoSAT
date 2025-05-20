# import logging
# from src.car_variables import CarType
# from src.control_modes.manual_mode.manual_mode import ManualMode
# from src.control_modes.autonomous_mode.autonomous_mode import AutonomousMode
#
#
# def main():
#     logging.basicConfig(level=logging.INFO)
#     mode = input("Which mode do you want to launch. 1 for manual, 2 for autonomous: ")
#     try:
#         car = int(input("Which car are we using: 0 for kart, 1 for hunter: "))
#         if car not in [0, 1]:
#             raise ValueError("Invalid car type. Please enter 0 for kart or 1 for hunter.")
#     except ValueError as e:
#         print(e)
#         return
#
#     if mode == "1":
#         manual = ManualMode(CarType(car))
#         manual.start()
#     elif mode == "2":
#         auto = AutonomousMode(CarType(car))
#         auto.start()
#
# if __name__ == "__main__":
#     main()


import logging
import uvicorn

from src.web_interface.app import app

def main():
    logging.basicConfig(level=logging.INFO)
    logging.info("Starting AutoSAT Control Panel")

    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()