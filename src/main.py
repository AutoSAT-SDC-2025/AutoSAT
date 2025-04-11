import logging
from src.car_variables import CarType
from src.control_modes.autonomous_mode.old_twente_code import go, new_go
from src.control_modes.manual_mode.manual_mode import ManualMode


def main():
    logging.basicConfig(level=logging.INFO)
    mode = input("Which mode do you want to launch. 1 for manual, 2 for autonomous: ")
    try:
        car = int(input("Which car are we using: 0 for kart, 1 for hunter: "))
        if car not in [0, 1]:
            raise ValueError("Invalid car type. Please enter 0 for kart or 1 for hunter.")
    except ValueError as e:
        print(e)
        return

    if mode == "1":
        manual = ManualMode(CarType(car))
        manual.start()
    elif mode == "2":
        new_go.main()
    # go.main()

if __name__ == "__main__":
    main()