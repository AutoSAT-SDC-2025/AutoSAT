import os
import cv2
import logging
import multiprocessing as mp

from src.util.video import get_camera_config, validate_camera_config
from src.multi_camera_calibration import CalibrationData, RenderDistance

WIDTH = 848
HEIGHT = 480

LineDetectionDims = {
    'width': 720,
    'height': 720
}


class RecordMode:
    def __init__(self, save_transforms: bool = False):
        self.save_transforms = save_transforms
        self.frame_save_dir = "capture_" + str(os.getpid())
        os.makedirs(self.frame_save_dir, exist_ok=True)

        self.frame_counter = 0
        self.cams = get_camera_config()
        self.captures = {}

        if self.save_transforms:
            self.data = CalibrationData(
                path="assets/calibration/latest.npz",
                input_shape=(1920, 1080),
                output_shape=(LineDetectionDims['width'], LineDetectionDims['height']),
                render_distance=RenderDistance(front=12.0, sides=6.0)
            )

    def setup_cameras(self):
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

            # Warm-up
            for _ in range(2):
                cap.read()

            self.captures[cam_name] = cap
            print(f"Camera {cam_name} ready.")

    def capture_and_save(self):
        frames = {}
        for cam_name, cap in self.captures.items():
            ret, frame = cap.read()
            if not ret:
                raise RuntimeError(f"Failed to read frame from {cam_name} camera.")
            frames[cam_name] = frame

            raw_path = f"{self.frame_save_dir}/frame_{self.frame_counter:05d}_{cam_name}.jpg"
            cv2.imwrite(raw_path, frame)

        if self.save_transforms:
            try:
                print("Stitching frames...")
                top_down = self.data.transform([frames['left'], frames['front'], frames['right']])
                stitched_path = f"{self.frame_save_dir}/frame_{self.frame_counter:05d}_stitched.jpg"
                cv2.imwrite(stitched_path, top_down)
            except Exception as e:
                logging.error(f"Stitching error: {e}")

        self.frame_counter += 1

    def start(self):
        logging.info("Starting record mode...")
        self.setup_cameras()

        try:
            while True:
                self.capture_and_save()
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        except Exception as e:
            logging.error(f"Error in record mode: {e}")
        finally:
            self.stop()
            cv2.destroyAllWindows()

    def stop(self):
        logging.info("Stopping and releasing cameras.")
        for cap in self.captures.values():
            cap.release()
        cv2.destroyAllWindows()
