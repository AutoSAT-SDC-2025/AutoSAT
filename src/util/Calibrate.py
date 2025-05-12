import cv2
from src.multi_camera_calibration import CalibrationPattern, Calibrator
from src.util.video import get_camera_config
import os

# Toggle this to switch between using cameras or loading static images
use_cameras = False


def calibrate_connected_cameras(save_path="./calibration"):
    captured_images = []

    if use_cameras:
        cams = get_camera_config()

        if not cams or not cams.get('left') or not cams.get('right') or not cams.get('front'):
            raise RuntimeError("No cameras connected or invalid camera configuration")

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

    else:
        # Load from assets/temp folder
        for cam_name in ['left', 'center', 'right']:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))  # Go from src/util/ -> AutoSAT
            img_path = os.path.join(project_root, "assets", "temp", f"{cam_name}.png")

            frame = cv2.imread(img_path)
            frame = cv2.resize(frame, (1920, 1080))

            if frame is None:
                raise RuntimeError(f"Failed to load image from {img_path}")

            captured_images.append(frame)

    pattern = CalibrationPattern(7, 5, 0.057, 0.043, cv2.aruco.DICT_4X4_100)


    calibrator = Calibrator(captured_images, pattern)
    calibrator.calibrate()
    calibrator.save(save_path, keep_history=False, overwrite=True)
    print(f"Calibration data saved to {save_path}")