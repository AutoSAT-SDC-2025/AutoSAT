import os
import cv2
import logging
import multiprocessing as mp
from datetime import datetime
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

        timestamp = datetime.now().strftime("%-d-%-m-%Y %H:%M")
        self.base_dir = os.path.join(".", timestamp)
        os.makedirs(self.base_dir, exist_ok=True)

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

    def capture_and_save(self, mode: str):
        frames = {}
        if mode in ("1", "2", "4"):
            required_cams = ['left', 'front', 'right']

            def capture_and_save(self, mode: str):
                iteration_dir = os.path.join(self.base_dir, str(self.frame_counter + 1))
                os.makedirs(iteration_dir, exist_ok=True)

                frames = {}
                if mode in ("1", "2", "4"):
                    required_cams = ['left', 'front', 'right']
                elif mode == "3":
                    required_cams = ['front']
                else:
                    logging.warning("Invalid mode, defaulting to all.")
                    required_cams = ['left', 'front', 'right']

                for cam_name in required_cams:
                    cap = self.captures.get(cam_name)
                    if not cap:
                        logging.error(f"Camera {cam_name} not available.")
                        continue

                    ret, frame = cap.read()
                    if not ret:
                        raise RuntimeError(f"Failed to read frame from {cam_name} camera.")
                    frames[cam_name] = frame

                    if mode in ("1", "2", "3"):
                        path = os.path.join(iteration_dir, f"frame_{self.frame_counter:05d}_{cam_name}.jpg")
                        cv2.imwrite(path, frame)

                if self.save_transforms and mode in ("1", "4") and all(k in frames for k in ['left', 'front', 'right']):
                    try:
                        print("Stitching frames...")
                        top_down = self.data.transform([frames['left'], frames['front'], frames['right']])
                        topdown_path = os.path.join(iteration_dir, f"frame_{self.frame_counter:05d}_topdown.jpg")
                        cv2.imwrite(topdown_path, top_down)

                        stitched_path = os.path.join(iteration_dir, f"frame_{self.frame_counter:05d}_stitched.jpg")
                        stitched = self.data.stitch([frames['left'], frames['front'], frames['right']])
                        cv2.imwrite(stitched_path, stitched)

                    except Exception as e:
                        logging.error(f"Stitching error: {e}")

                self.frame_counter += 1
                print(f"Captured frame {self.frame_counter} for mode {mode}")

    def start(self):
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
