import os
import cv2
import glob

from src.util.video import get_camera_config
from ..control_modes.autonomous_mode.object_detection.ObjectDetection import ObjectDetection
from ..control_modes.autonomous_mode.object_detection.TrafficDetection import TrafficManager
from ..control_modes.autonomous_mode.line_detection.LineDetection import LineFollowingNavigation
from .Render import Renderer

WIDTH = 848
HEIGHT = 480


def load_image_paths(folder_path):
    return sorted(glob.glob(os.path.join(folder_path, '*.png')) +
                  glob.glob(os.path.join(folder_path, '*.jpg')) +
                  glob.glob(os.path.join(folder_path, '*.jpeg')))


def main(weights_path: str, input_source: str, image_folder: str):
    print("Starting image viewer...")
    image_paths = load_image_paths(image_folder)
    if not image_paths:
        print(f"No images found in {image_folder}")
        return

    index = 0
    total_images = len(image_paths)
    playing = False  # Auto-play state

    object_detector = ObjectDetection(weights_path, input_source)
    traffic_manager = TrafficManager()
    line_detection = LineFollowingNavigation(width=WIDTH, height=HEIGHT)
    renderer = Renderer()

    while True:
        frame = cv2.imread(image_paths[index])
        if frame is None:
            print(f"Could not read image: {image_paths[index]}")
            break
        frame = cv2.resize(frame, (WIDTH, HEIGHT))
        print("Processing image:", image_paths[index])

        renderer.clear()
        traffic_state, detections, object_visuals = object_detector.process(frame)
        renderer.add_drawings(object_visuals)

        # Uncomment if you want line detection visuals
        steering_angle, speed, line_visuals = line_detection.process(frame)
        renderer.add_drawings(line_visuals)

        renderer.render(frame)

        # Auto-play: wait 20ms and advance, else wait indefinitely
        key = cv2.waitKey(20 if playing else 0)

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
        elif key == ord('x'):  # Skip forward 50
            index = (index + 50) % total_images
            print("Skipped forward 50 frames:", image_paths[index])
        elif key == ord('z'):  # Skip back 50
            index = (index - 50) % total_images
            print("Skipped back 50 frames:", image_paths[index])
        elif playing:
            index = (index + 1) % total_images

        print("Current image index:", index, "/", total_images)

    cv2.destroyAllWindows()


if __name__ == '__main__':

    path = 'D:/gebruiker/Desktop/Test-Data/'
    # ask for which camera (stitched, front, topdown, left, right)
    inp = input("Enter camera type (stitched, front, topdown, left, right): ").strip().lower()
    if inp not in ['stitched', 'front', 'topdown', 'left', 'right']:
        print("Invalid camera type. Defaulting to stitched.")
        inp = 'stitched'

   #todo: kalman checkout
    main('assets/v5_model.pt', 'images', path + inp + '/')