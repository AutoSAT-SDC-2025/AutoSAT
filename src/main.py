import logging

from src.car_variables import CarType
from src.control_modes.autonomous_mode.autonomous_mode import AutonomousMode
from src.control_modes.manual_mode.manual_mode import ManualMode
from src.control_modes.recording_mode.recording_mode import RecordMode
# from src.car_variables import CarType
# from src.control_modes.manual_mode.manual_mode import ManualMode
# from src.control_modes.autonomous_mode.autonomous_mode import AutonomousMode
from src.util.Calibrate import calibrate_connected_cameras, transform_camera_image

def main():
    logging.basicConfig(level=logging.INFO)
    mode = input("Which mode do you want to launch. 1 for manual, 2 for autonomous, 3 for recording, 4 to calibrate, 5 to pray: ")
    try:
        if mode not in ["4"] and mode not in ["5"]: # maybe more later.
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
        record = RecordMode(CarType(car))
        record.start()

    elif mode == "4":
        calibrate_connected_cameras(save_path="assets/calibration")
    # go.main()

    elif mode == "5":
        transform_camera_image()

if __name__ == "__main__":
    main()