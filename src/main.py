import logging
from src.car_variables import CarType
from src.control_modes.manual_mode.manual_mode import ManualMode
from src.control_modes.autonomous_mode.autonomous_mode import AutonomousMode
from src.util.Calibrate import calibrate_connected_cameras

def main():
    logging.basicConfig(level=logging.INFO)
    mode = input("Which mode do you want to launch. 1 for manual, 2 for autonomous, 3 to calibrate")
    try:
        if mode not in ["3"]: # maybe more later.
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
        auto = AutonomousMode(CarType(car))
        auto.start()
    elif mode == "3":
        calibrate_connected_cameras(save_path="../assets/calibration")
    # go.main()

if __name__ == "__main__":
    main()