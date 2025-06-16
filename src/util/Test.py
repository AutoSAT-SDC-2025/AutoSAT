import os
import cv2
import glob

from src.util.video import get_camera_config
from ..control_modes.autonomous_mode.object_detection.ObjectDetection import ObjectDetection
from ..control_modes.autonomous_mode.object_detection.TrafficDetection import TrafficManager
from ..control_modes.autonomous_mode.line_detection.LineDetection import LineFollowingNavigation
from .Render import Renderer

# Image dimensions for resizing input frames
WIDTH = 848
HEIGHT = 480

def load_image_paths(folder_path):
    """
    Loads and returns a sorted list of image file paths (png, jpg, jpeg) from the given folder.

    Args:
        folder_path (str): Path to the folder containing images.

    Returns:
        list: Sorted list of image file paths.
    """
    return sorted(glob.glob(os.path.join(folder_path, '*.png')) +
                  glob.glob(os.path.join(folder_path, '*.jpg')) +
                  glob.glob(os.path.join(folder_path, '*.jpeg')))

def main(weights_path: str, input_source: str, image_folder: str):
    """
    Main function for interactive image viewer and detection tester.
    Loads images from a folder, runs object and line detection, and displays results with navigation controls.

    Args:
        weights_path (str): Path to the object detection model weights.
        input_source (str): Input source type (e.g., 'images').
        image_folder (str): Folder containing images to process.
    """
    print("Starting image viewer...")
    image_paths = load_image_paths(image_folder)
    if not image_paths:
        print(f"No images found in {image_folder}")
        return

    index = 0
    total_images = len(image_paths)
    playing = False  # Auto-play state

    # Initialize detection and rendering modules
    object_detector = ObjectDetection(weights_path, input_source)
    traffic_manager = TrafficManager()
    line_detection = LineFollowingNavigation(width=WIDTH, height=HEIGHT, mode='normal')
    renderer = Renderer()

    while True:
        # Load and resize the current image
        frame = cv2.imread(image_paths[index])
        if frame is None:
            print(f"Could not read image: {image_paths[index]}")
            break
        frame = cv2.resize(frame, (WIDTH, HEIGHT))
        print("Processing image:", image_paths[index])

        renderer.clear()  # Clear previous drawing operations

        # Run object detection and get drawing operations for visualization
        traffic_state, detections, object_visuals = object_detector.process(frame)
        renderer.add_drawings_objectdetection(object_visuals)

        # Placeholder for further per-detection processing if needed
        for detection in detections:
            pass

        # Run line detection and get drawing operations for visualization
        steering_angle, speed, line_visuals = line_detection.process(frame)
        renderer.add_drawings_linedetection(line_visuals)
        renderer.render_lines(frame)
        new_frame = renderer.get_last_linedetection_image()

        # Display the processed frame with overlays
        cv2.imshow("Frame", new_frame)

        # Handle keyboard input for navigation and playback
        key = cv2.waitKey(20 if playing else 0)

        print("Suggested steering angle:", steering_angle)

        if key == ord('q') or key == 27:  # Quit on 'q' or ESC
            break
        elif key == ord('a'):  # Previous image
            index = (index - 1) % total_images
            print("Previous image:", image_paths[index])
        elif key == ord('d'):  # Next image
            index = (index + 1) % total_images
            print("Next image:", image_paths[index])
        elif key == ord(' '):  # Toggle auto-play
            playing = not playing
            print("Auto-play:", "ON" if playing else "OFF")
        elif key == ord('x'):  # Skip forward 50 images
            index = (index + 50) % total_images
            print("Skipped forward 50 frames:", image_paths[index])
        elif key == ord('z'):  # Skip back 50 images
            index = (index - 50) % total_images
            print("Skipped back 50 frames:", image_paths[index])
        elif playing:
            index = (index + 1) % total_images

        print("Current image index:", index, "/", total_images)

    cv2.destroyAllWindows()

if __name__ == '__main__':
    # Determine script directory for relative path resolution
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Prompt user for camera type (subfolder in assets/data/)
    inp = input("Enter camera type (stitched, front, topdown, left, right): ").strip().lower()
    if inp not in ['stitched', 'front', 'topdown', 'left', 'right']:
        print("Invalid camera type. Defaulting to stitched.")
        inp = 'stitched'

    # Compute project root and image/model paths
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
    image_folder = os.path.join('assets/data/', inp)
    weights_path = os.path.join(project_root, 'assets', 'v5_model.pt')

    print("Image folder:", image_folder)
    main(weights_path, 'images', image_folder)