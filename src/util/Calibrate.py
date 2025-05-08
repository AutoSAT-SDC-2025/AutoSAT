import cv2
from src.multi_camera_calibration import CalibrationPattern, Calibrator
from src.util.video import get_camera_config


def calibrate_connected_cameras(save_path="./calibration"):
    cams = get_camera_config()

    if not cams or not cams.get('left') or not cams.get('right') or not cams.get('front'):
        raise RuntimeError("No cameras connected or invalid camera configuration")

    captured_images = []

    for cam_name in ['left', 'front', 'right']:
        cam_path = cams[cam_name]
        cap = cv2.VideoCapture(cam_path)

        if not cap.isOpened():
            raise RuntimeError(f"Failed to open {cam_name} camera at {cam_path}")

        for _ in range(3):
            cap.read()

        ret, frame = cap.read()
        cap.release()

        if not ret:
            raise RuntimeError(f"Failed to read frame from {cam_name} camera at {cam_path}")

        captured_images.append(frame)

    pattern = CalibrationPattern(
        width=10,
        height=8,
        square_length=0.115,
        marker_length=0.086,
        aruco_dict=cv2.aruco.DICT_4X4_100
    )

    calibrator = Calibrator(captured_images, pattern)
    calibrator.calibrate()
    calibrator.save(save_path, keep_history=False)
