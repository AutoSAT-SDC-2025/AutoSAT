import logging
import time
import glob
import numpy as np
from typing import Dict
import cv2
import os
from src.car_variables import CameraResolution, LineDetectionDims
from src.multi_camera_calibration import CalibrationData, RenderDistance
from src.util.video import get_camera_config, validate_camera_config

class CameraController:

    def __init__(self):
        self.__cameras: Dict[str, cv2.VideoCapture] = dict()
        self.__camera_frames = {}
        self.__enable = False
        
        try:
            self.__camera_paths = get_camera_config()
            if validate_camera_config(self.__camera_paths):
                self.__using_fallback = False
                self.__calibration_data = CalibrationData(
                    path="assets/calibration/latest.npz",
                    input_shape=(1920, 1080),
                    output_shape=(LineDetectionDims.WIDTH, LineDetectionDims.HEIGHT),
                    render_distance=RenderDistance(front=12.0, sides=6.0)
                )
            else:
                raise RuntimeError("No cameras connected")
        except Exception as e:
            logging.warning(f"No cameras available: {e}. Using recorded session images.")
            self.__using_fallback = True
            self.__setup_fallback_images()

    def __setup_fallback_images(self):
        self.__image_paths = {"front": [], "left": [], "right": [], "stitched": [], "topdown": []}
        self.__image_index = {"front": 0, "left": 0, "right": 0, "stitched": 0, "topdown": 0}
        
        RECORDED_SESSION = "session_2025-05-22_15-19-22_266"
        
        base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "logs")
        if not os.path.exists(base_dir):
            logging.error(f"Logs directory not found: {base_dir}")
            return
            
        session_path = os.path.join(base_dir, RECORDED_SESSION)
        images_path = os.path.join(session_path, "images")
        
        if not os.path.exists(images_path):
            logging.warning(f"Recorded session path not found: {images_path}")
            logging.warning(f"Falling back to automatic session detection")
            session_dirs = sorted([d for d in os.listdir(base_dir) if d.startswith("session_")], reverse=True)
            
            for session in session_dirs:
                session_path = os.path.join(base_dir, session)
                images_path = os.path.join(session_path, "images")
                
                if os.path.exists(images_path) and self.__has_valid_images(images_path):
                    logging.info(f"Using fallback images from session: {session}")
                    break
            else:
                logging.error("No suitable log sessions with images found for fallback")
                return
        else:
            logging.info(f"Using recorded session: {RECORDED_SESSION}")
        
        self.__load_images_from_session(images_path)

    def __has_valid_images(self, images_path):
        for view in self.__image_paths.keys():
            view_path = os.path.join(images_path, view)
            if os.path.exists(view_path) and any(f.endswith('.jpg') for f in os.listdir(view_path)):
                return True
        return False

    def __load_images_from_session(self, images_path):
        for view in self.__image_paths.keys():
            view_path = os.path.join(images_path, view)
            if os.path.exists(view_path):
                image_files = sorted(glob.glob(os.path.join(view_path, "*.jpg")))
                if image_files:
                    self.__image_paths[view] = image_files
                    logging.info(f"Loaded {len(image_files)} {view} images")
        
        self.__preload_frames()

    def __preload_frames(self):
        for view, paths in self.__image_paths.items():
            if paths:
                frame = cv2.imread(paths[0])
                self.__camera_frames[view] = frame if frame is not None else self.__create_blank_frame()
            else:
                self.__camera_frames[view] = self.__create_blank_frame()

    def __create_blank_frame(self):
        frame = np.zeros((CameraResolution.HEIGHT, CameraResolution.WIDTH, 3), dtype=np.uint8)
        cv2.putText(frame, "No image available", (50, CameraResolution.HEIGHT // 2), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        return frame

    def setup_cameras(self):
        if self.__using_fallback:
            logging.info("Using recorded session images - skipping camera setup")
            return
            
        if not hasattr(self, "_CameraController__camera_paths") or not self.__camera_paths:
            logging.error("No camera paths available for setup")
            return
            
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

    def enable_cameras(self):
        self.__enable = True

    def disable_cameras(self):
        self.__enable = False

    def capture_camera_frames(self):
        if not self.__enable:
            return
            
        if self.__using_fallback:
            for view, paths in self.__image_paths.items():
                if paths:
                    self.__image_index[view] = (self.__image_index[view] + 1) % len(paths)
                    frame = cv2.imread(paths[self.__image_index[view]])
                    if frame is not None:
                        self.__camera_frames[view] = frame
            time.sleep(0.033)
        else:
            for cam_name, cap in self.__cameras.items():
                ret, frame = cap.read()
                if ret:
                    self.__camera_frames[cam_name] = frame

    def get_front_view(self):
        return self.__camera_frames.get('front', self.__create_blank_frame())

    def get_left_view(self):
        return self.__camera_frames.get('left', self.__create_blank_frame())

    def get_right_view(self):
        return self.__camera_frames.get('right', self.__create_blank_frame())

    def get_top_down_view(self):
        if self.__using_fallback:
            return self.__camera_frames.get('topdown', self.__create_blank_frame())
        else:
            return self.__calibration_data.transform([
                self.__camera_frames.get('left', self.__create_blank_frame()),
                self.__camera_frames.get('front', self.__create_blank_frame()),
                self.__camera_frames.get('right', self.__create_blank_frame())
            ])

    def get_stitched_image(self):
        if self.__using_fallback:
            return self.__camera_frames.get('stitched', self.__create_blank_frame())
        else:
            return self.__calibration_data.stitch([
                self.__camera_frames.get('left', self.__create_blank_frame()),
                self.__camera_frames.get('front', self.__create_blank_frame()),
                self.__camera_frames.get('right', self.__create_blank_frame())
            ])

def return_lower_rez(frame):
    return cv2.resize(frame, (CameraResolution.WIDTH, CameraResolution.HEIGHT))