"""
Camera controller module for autonomous vehicle camera management.

This module provides the CameraController class which handles multiple camera
operations including live camera capture, fallback to recorded images, and
camera calibration for different vehicle types.
"""

import logging
import time
import glob
import numpy as np
from typing import Dict
import cv2
import os
from src.car_variables import CameraResolution, LineDetectionDims, CarType
from src.multi_camera_calibration import CalibrationData, RenderDistance
from src.util.video import get_camera_config, validate_camera_config

class CameraController:
    """
    Controls camera operations for the autonomous vehicle system.
    
    This class manages multiple cameras (front, left, right), handles image capture,
    and provides fallback functionality using recorded session images when cameras
    are not available. It also handles camera calibration based on vehicle type.
    
    Attributes:
        __cameras: Dictionary of active camera captures indexed by camera position
        __camera_frames: Current frames from each camera
        __enable: Whether cameras are currently enabled for capture
        __car_type: Type of vehicle (kart or hunter) for calibration
        __using_fallback: Whether using recorded images instead of live cameras
        __calibration_data: Camera calibration data for coordinate transformations
    """

    def __init__(self):
        """
        Initialize the camera controller.
        
        Sets up camera paths, calibration data, and falls back to recorded
        images if no cameras are available.
        
        Raises:
            RuntimeError: If no cameras are connected and fallback setup fails
        """
        self.__cameras: Dict[str, cv2.VideoCapture] = dict()
        self.__camera_frames = {}
        self.__enable = False
        self.__car_type = None
        
        try:
            self.__camera_paths = get_camera_config()
            if validate_camera_config(self.__camera_paths):
                self.__using_fallback = False
                if self.__car_type is CarType.kart:
                    self.__calibration_data = CalibrationData(
                        path="assets/calibration/kart_calib.npz",
                        input_shape=(1920, 1080),
                        output_shape=(LineDetectionDims.WIDTH, LineDetectionDims.HEIGHT),
                        render_distance=RenderDistance(front=12.0, sides=6.0)
                    )
                elif self.__car_type is CarType.hunter:
                    self.__calibration_data = CalibrationData(
                        path="assets/calibration/hunter_calib.npz",
                        input_shape=(1920, 1080),
                        output_shape=(LineDetectionDims.WIDTH, LineDetectionDims.HEIGHT),
                        render_distance=RenderDistance(front=12.0, sides=6.0)
                    )
                else:
                    self.__calibration_data = CalibrationData(
                        path="assets/calibration/kart_calib.npz",
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

    def set_car_type(self, car_type: CarType):
        """
        Set the vehicle type for calibration purposes.
        
        Args:
            car_type: The type of vehicle (CarType.kart or CarType.hunter)
        """
        self.__car_type = car_type

    def __setup_fallback_images(self):
        """
        Set up fallback images from recorded sessions.
        
        Searches for recorded session directories and loads image sequences
        for each camera view. Falls back to automatic session detection if
        the default session is not found.
        """
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
        """
        Check if a session directory contains valid images.
        
        Args:
            images_path: Path to the images directory
            
        Returns:
            True if the directory contains valid JPG images, False otherwise
        """
        for view in self.__image_paths.keys():
            view_path = os.path.join(images_path, view)
            if os.path.exists(view_path) and any(f.endswith('.jpg') for f in os.listdir(view_path)):
                return True
        return False

    def __load_images_from_session(self, images_path):
        """
        Load image sequences from a recorded session.
        
        Args:
            images_path: Path to the session images directory
        """
        for view in self.__image_paths.keys():
            view_path = os.path.join(images_path, view)
            if os.path.exists(view_path):
                image_files = sorted(glob.glob(os.path.join(view_path, "*.jpg")))
                if image_files:
                    self.__image_paths[view] = image_files
                    logging.info(f"Loaded {len(image_files)} {view} images")
        
        self.__preload_frames()

    def __preload_frames(self):
        """
        Preload the first frame from each camera view.
        
        Loads the first image from each view's image sequence to initialize
        the camera frames dictionary.
        """
        for view, paths in self.__image_paths.items():
            if paths:
                frame = cv2.imread(paths[0])
                self.__camera_frames[view] = frame if frame is not None else self.__create_blank_frame()
            else:
                self.__camera_frames[view] = self.__create_blank_frame()

    def __create_blank_frame(self):
        """
        Create a blank frame with "No image available" text.
        
        Creates a black frame with appropriate dimensions and displays
        an error message in white text.
        """
        frame = np.zeros((CameraResolution.HEIGHT, CameraResolution.WIDTH, 3), dtype=np.uint8)
        cv2.putText(frame, "No image available", (50, CameraResolution.HEIGHT // 2), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        return frame

    def setup_cameras(self):
        """
        Initialize and configure all connected cameras.
        
        Sets camera properties including resolution (1920x1080), FPS (30),
        codec (MJPG), and focus settings. Skips setup if using fallback mode.
        
        Raises:
            RuntimeError: If any camera fails to open
        """
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
        """Enable camera frame capture."""
        self.__enable = True

    def disable_cameras(self):
        """Disable camera frame capture."""
        self.__enable = False

    def capture_camera_frames(self):
        """
        Capture frames from all enabled cameras.
        
        Updates internal frame storage with latest camera data. If using fallback
        mode, cycles through recorded images with a 33ms delay to simulate real-time.
        If using live cameras, captures frames from all active cameras.
        """
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
        """
        Get the current front camera frame.
        
        Returns the front camera frame or a blank frame if unavailable.
        """
        return self.__camera_frames.get('front', self.__create_blank_frame())

    def get_left_view(self):
        """
        Get the current left camera frame.
        
        Returns the left camera frame or a blank frame if unavailable.
        """
        return self.__camera_frames.get('left', self.__create_blank_frame())

    def get_right_view(self):
        """
        Get the current right camera frame.
        
        Returns the right camera frame or a blank frame if unavailable.
        """
        return self.__camera_frames.get('right', self.__create_blank_frame())

    def get_top_down_view(self):
        """
        Get the top-down transformed view.
        
        In fallback mode, returns the recorded top-down image. In live mode,
        applies calibration transformation to combine left, front, and right
        camera views into a bird's-eye perspective.
        """
        if self.__using_fallback:
            return self.__camera_frames.get('topdown', self.__create_blank_frame())
        else:
            return self.__calibration_data.transform([
                self.__camera_frames.get('left', self.__create_blank_frame()),
                self.__camera_frames.get('front', self.__create_blank_frame()),
                self.__camera_frames.get('right', self.__create_blank_frame())
            ])

    def get_stitched_image(self):
        """
        Get the panoramic stitched view.
        
        In fallback mode, returns the recorded stitched image. In live mode,
        applies calibration stitching to combine left, front, and right
        camera views into a panoramic view.
        """
        if self.__using_fallback:
            return self.__camera_frames.get('stitched', self.__create_blank_frame())
        else:
            return self.__calibration_data.stitch([
                self.__camera_frames.get('left', self.__create_blank_frame()),
                self.__camera_frames.get('front', self.__create_blank_frame()),
                self.__camera_frames.get('right', self.__create_blank_frame())
            ])

def return_lower_rez(frame):
    """
    Resize frame to lower resolution.
    
    Takes an input frame and resizes it to the dimensions specified
    in CameraResolution configuration.
    
    Args:
        frame: Input frame to resize
    """
    return cv2.resize(frame, (CameraResolution.WIDTH, CameraResolution.HEIGHT))