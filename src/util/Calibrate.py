
import cv2
from src.multi_camera_calibration import CalibrationPattern, Calibrator, CalibrationData, RenderDistance
from src.util.video import get_camera_config
import os

# This script is used for calibrating the multi-camera system and for testing the calibration results.
# It can capture images from connected cameras or load static images, perform calibration, and visualize the transformation.

# Toggle this to switch between using live cameras or loading static images for calibration/testing
use_cameras = True

def calibrate_connected_cameras(save_path="./calibration"):
    """
    Captures images from all connected cameras (or loads static images), 
    detects the calibration pattern, and computes camera calibration parameters.
    The calibration data is then saved to disk for later use.

    Args:
        save_path (str): Directory where calibration data will be saved.
    """
    captured_images = []

    if use_cameras:
        # Get camera configuration (paths for left, front, right cameras)
        cams = get_camera_config()

        # Ensure all required cameras are available
        if not cams or not cams.get('left') or not cams.get('right') or not cams.get('front'):
            raise RuntimeError("No cameras connected or invalid camera configuration")

        # Capture a single frame from each camera after warming up
        for cam_name in ['left', 'front', 'right']:
            cam_path = cams[cam_name]
            cap = cv2.VideoCapture(cam_path)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

            if not cap.isOpened():
                raise RuntimeError(f"Failed to open {cam_name} camera at {cam_path}")

            # Warm up the camera to stabilize exposure/white balance
            for _ in range(3):
                cap.read()

            ret, frame = cap.read()
            cap.release()

            if not ret:
                raise RuntimeError(f"Failed to read frame from {cam_name} camera at {cam_path}")

            captured_images.append(frame)

    else:
        # Load static images from assets/temp for calibration (useful for offline or repeatable calibration)
        for cam_name in ['left', 'center', 'right']:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))  # Go from src/util/ -> project root
            img_path = os.path.join(project_root, "assets", "temp", f"{cam_name}.png")

            frame = cv2.imread(img_path)
            frame = cv2.resize(frame, (1920, 1080))

            if frame is None:
                raise RuntimeError(f"Failed to load image from {img_path}")

            captured_images.append(frame)

    # Define the calibration pattern (e.g., ArUco board with specific dimensions)
    pattern = CalibrationPattern(10, 7, 0.115, 0.085, cv2.aruco.DICT_4X4_100)

    # Create calibrator, run calibration, and save results
    calibrator = Calibrator(captured_images, pattern)
    calibrator.calibrate()
    calibrator.save(save_path, keep_history=False, overwrite=True)
    print(f"Calibration data saved to {save_path}")

def transform_camera_image():
    """
    Loads the latest calibration data and applies the transformation to new images
    captured from the cameras. This function is used to test and visualize the calibration
    by displaying the stitched and top-down transformed images.
    """
    # Load calibration data from file
    data = CalibrationData(
        path="assets/calibration/latest.npz",
        input_shape=(1920, 1080),
        output_shape=(720, 720),
        render_distance=RenderDistance(
            front=12.0,
            sides=6.0
        )
    )

    captured_images = []

    if use_cameras:
        # Get camera configuration and capture a frame from each camera
        cams = get_camera_config()

        if not cams or not cams.get('left') or not cams.get('right') or not cams.get('front'):
            raise RuntimeError("No cameras connected or invalid camera configuration")

        for cam_name in ['left', 'front', 'right']:
            cam_path = cams[cam_name]
            cap = cv2.VideoCapture(cam_path)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

            if not cap.isOpened():
                raise RuntimeError(f"Failed to open {cam_name} camera at {cam_path}")

            # Warm up the camera
            for _ in range(3):
                cap.read()

            ret, frame = cap.read()
            # Optionally, show the frame for debugging
            # cv2.imshow("please god work",frame)
            # cv2.waitKey(0)
            cap.release()

            if not ret:
                raise RuntimeError(f"Failed to read frame from {cam_name} camera at {cam_path}")

            captured_images.append(frame)

    # Apply the calibration: stitch and transform the images
    stitched = data.stitch(captured_images)
    topdown = data.transform(captured_images)

    # Display the results for visual inspection of calibration quality
    cv2.imshow("Stitched", cv2.resize(stitched, (720, 720 * stitched.shape[0] // stitched.shape[1])))
    cv2.imshow("Top-Down", topdown)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    """
    Main entry point: runs the calibration process.
    """
    calibrate_connected_cameras()

if __name__ == "__main__":
    main()