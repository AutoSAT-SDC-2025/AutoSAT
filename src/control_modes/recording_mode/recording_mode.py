# NOTE: This is our previously used recorder.
# We are now using our cameramanager for recording and camera management.

import os
import cv2
import logging
import multiprocessing as mp
from datetime import datetime
from src.util.video import get_camera_config, validate_camera_config
from src.multi_camera_calibration import CalibrationData, RenderDistance

# Constants for image dimensions used in recording and line detection
WIDTH = 848
HEIGHT = 480

LineDetectionDims = {
    'width': 720,
    'height': 720
}

class RecordMode:
    """
    Handles multi-camera video recording, frame capture, and optional transformation/stitching.
    This class was previously used for recording, but has been superseded by the cameramanager.
    """
    def __init__(self, save_transforms: bool = False):
        """
        Initialize the recording mode, set up directories, and prepare calibration if needed.

        Args:
            save_transforms (bool): If True, perform and save top-down and stitched transforms.
        """
        self.save_transforms = save_transforms

        # Create a unique directory for this recording session, using a timestamp
        timestamp = datetime.now().strftime("%-d-%-m-%Y %H:%M")
        self.base_dir = os.path.join("recorded", timestamp)
        os.makedirs(self.base_dir, exist_ok=True)

        self.frame_counter = 0  # Counts the number of frames captured
        self.cams = get_camera_config()  # Load camera paths/configuration
        self.captures = {}  # Will hold cv2.VideoCapture objects for each camera

        # If transforms are to be saved, load calibration data for perspective transforms
        if self.save_transforms:
            self.data = CalibrationData(
                path="assets/calibration/latest.npz",
                input_shape=(1920, 1080),
                output_shape=(LineDetectionDims['width'], LineDetectionDims['height']),
                render_distance=RenderDistance(front=12.0, sides=6.0)
            )

    def setup_cameras(self):
        """
        Initialize and open all required cameras, set resolution, and warm up.
        Raises an error if any camera is missing or fails to open.
        """
        for cam_name in ['left', 'front', 'right']:
            print(f"Setting up camera: {cam_name}")
            cam_path = self.cams.get(cam_name)
            if not cam_path:
                raise RuntimeError(f"Missing camera path for {cam_name}")

            cap = cv2.VideoCapture(cam_path)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

            if not cap.isOpened():
                raise RuntimeError(f"Failed to open {cam_name} camera at {cam_path}")

            # Warm-up: read a couple of frames to stabilize exposure/white balance
            for _ in range(2):
                cap.read()

            self.captures[cam_name] = cap
            print(f"Camera {cam_name} ready.")

    def capture_and_save(self, mode: str):
        """
        Capture frames from the selected cameras and save them to disk.
        Optionally performs top-down and stitched transforms if enabled.

        Args:
            mode (str): Capture mode ('1', '2', '3', or '4') determining which cameras/outputs to save.
        """
        # Create a new directory for this frame iteration
        iteration_dir = os.path.join(self.base_dir, str(self.frame_counter + 1))
        os.makedirs(iteration_dir, exist_ok=True)

        frames = {}
        # Determine which cameras are required based on the selected mode
        if mode in ("1", "2", "4"):
            required_cams = ['left', 'front', 'right']
        elif mode == "3":
            required_cams = ['front']
        else:
            logging.warning("Invalid mode, defaulting to all.")
            required_cams = ['left', 'front', 'right']

        # Capture a frame from each required camera
        for cam_name in required_cams:
            cap = self.captures.get(cam_name)
            if not cap:
                logging.error(f"Camera {cam_name} not available.")
                continue

            ret, frame = cap.read()
            if not ret:
                raise RuntimeError(f"Failed to read frame from {cam_name} camera.")
            frames[cam_name] = frame

            # Save individual camera frames if mode requires it
            if mode in ("1", "2", "3"):
                path = os.path.join(iteration_dir, f"frame_{self.frame_counter:05d}_{cam_name}.jpg")
                cv2.imwrite(path, frame)

        # If transforms are enabled and all main cameras are available, save top-down and stitched images
        if self.save_transforms and mode in ("1", "4") and all(k in frames for k in ['left', 'front', 'right']):
            try:
                print("Stitching frames...")
                # Top-down transformation
                top_down = self.data.transform([frames['left'], frames['front'], frames['right']])
                topdown_path = os.path.join(iteration_dir, f"frame_{self.frame_counter:05d}_topdown.jpg")
                cv2.imwrite(topdown_path, top_down)

                # Stitched panorama
                stitched_path = os.path.join(iteration_dir, f"frame_{self.frame_counter:05d}_stitched.jpg")
                stitched = self.data.stitch([frames['left'], frames['front'], frames['right']])
                cv2.imwrite(stitched_path, stitched)

            except Exception as e:
                logging.error(f"Stitching error: {e}")

        self.frame_counter += 1
        print(f"Captured frame {self.frame_counter} for mode {mode}")

    def start(self):
        """
        Main entry point for recording mode.
        Prompts user for capture mode, sets up cameras, and enters capture loop.
        Exits on 'q' key or error.
        """
        print("Select capture mode:")
        print("1 = all (left, front, right + stitched)")
        print("2 = only 3 main cams (left, front, right)")
        print("3 = only front camera")
        print("4 = stitched only (requires all cameras)")
        mode = input("Enter mode [1-4]: ").strip()

        self.setup_cameras()

        try:
            logging.info("Starting record mode...")
            while True:
                self.capture_and_save(mode)
                # Wait for 'q' key to exit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        except Exception as e:
            logging.error(f"Error in record mode: {e}")
        finally:
            self.stop()
            cv2.destroyAllWindows()

    def stop(self):
        """
        Release all camera resources and close OpenCV windows.
        """
        logging.info("Stopping and releasing cameras.")
        for cap in self.captures.values():
            cap.release()
        cv2.destroyAllWindows()