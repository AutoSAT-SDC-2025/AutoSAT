import cv2
from src.multi_camera_calibration import CalibrationPattern, Calibrator

def calibrate_connected_cameras(save_path="./calibration", max_devices=5):
    def get_connected_cameras(max_devices):
        cameras = []
        for idx in range(max_devices):
            cap = cv2.VideoCapture(idx)
            if cap.read()[0]:
                cameras.append(idx)
            cap.release()
        return cameras

    camera_indices = get_connected_cameras(max_devices)

    if len(camera_indices) < 3:
        raise RuntimeError("Need at least 3 connected cameras.")

    captured_images = []
    for idx in camera_indices[:3]:
        cap = cv2.VideoCapture(idx)
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError(f"Failed to read from camera {idx}")
        captured_images.append(frame)
        cap.release()

    pattern = CalibrationPattern(10, 8, 0.115, 0.086, cv2.aruco.DICT_4X4_100)

    calibrator = Calibrator(captured_images, pattern)
    calibrator.calibrate()
    calibrator.save(save_path, keep_history=False)
