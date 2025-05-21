import logging
from typing import Dict
from src.car_variables import CameraResolution, LineDetectionDims
from src.multi_camera_calibration import CalibrationData, RenderDistance
from src.util.video import get_camera_config, validate_camera_config
import cv2

class CameraController:

    def __init__(self):
        self.__cameras : Dict[str, cv2.VideoCapture] = dict()
        self.__camera_paths = get_camera_config()
        if not validate_camera_config(self.__camera_paths):
            raise RuntimeError("No cameras connected or invalid camera configuration")
        self.__camera_frames = {}
        self.__calibration_data = CalibrationData(
            path="assets/calibration/latest.npz",
            input_shape=(1920, 1080),
            output_shape=(LineDetectionDims.WIDTH, LineDetectionDims.HEIGHT),
            render_distance=RenderDistance(
                front=12.0,
                sides=6.0
            )
        )
        self.__enable = False

    def setup_cameras(self):
        for camera_type, path in self.__camera_paths.items():
            capture = cv2.VideoCapture(path)
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            capture.set(cv2.CAP_PROP_FOCUS, 0)
            capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))
            capture.set(cv2.CAP_PROP_FPS, 30)

            if not capture.isOpened():
                raise RuntimeError(f"Failed to open {camera_type} camera at {path}")

            self.__cameras[camera_type] = capture
            logging.info(f"Completed camera setup for {camera_type} at {path}")

    def enable_cameras(self):
        self.__enable = True

    def disable_cameras(self):
        self.__enable = False

    def capture_camera_frames(self):
        if self.__enable:
            for cam_name, cap in self.__cameras.items():
                ret, frame = cap.read()
                if not ret:
                    raise RuntimeError(f"Failed to read frame from {cam_name} camera.")
                self.__camera_frames[cam_name] = frame
            return True
        return False

    def get_front_view(self):
        return self.__camera_frames['front']

    def get_left_view(self):
        return self.__camera_frames['left']

    def get_right_view(self):
        return self.__camera_frames['right']

    def get_top_down_view(self):
        return  self.__calibration_data.transform([
            self.__camera_frames['left'], 
            self.__camera_frames['front'], 
            self.__camera_frames['right']
        ])

    def get_stitched_image(self):
        return  self.__calibration_data.stitch([
            self.__camera_frames['left'],
            self.__camera_frames['front'], 
            self.__camera_frames['right']
        ])

def return_lower_rez(frame):
    return cv2.resize(frame, (CameraResolution.WIDTH, CameraResolution.HEIGHT))