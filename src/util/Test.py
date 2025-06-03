import os

import cv2

from src.util.video import get_camera_config
from ..control_modes.autonomous_mode.object_detection.ObjectDetection import ObjectDetection
from ..control_modes.autonomous_mode.object_detection.TrafficDetection import TrafficManager

from ..control_modes.autonomous_mode.line_detection.LineDetection import LineFollowingNavigation

from .Render import Renderer

WIDTH = 848
HEIGHT = 480

def main(weights_path: str, input_source: str, video_path: str = None):
    print("Starting test script...")
    object_detector = ObjectDetection(weights_path, input_source)
    print("Object detector initialized.")
    traffic_manager = TrafficManager()
    line_detection = LineFollowingNavigation(width=WIDTH, height=HEIGHT)
    renderer = Renderer()
    print("Object detector and line detection initialized.")

    # cap = cv2.VideoCapture(2)
    cap = cv2.VideoCapture("D:\\gebruiker\\Downloads\\Car.mp4")

    #cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        renderer.clear()

        traffic_state, detections, object_visuals = object_detector.process(frame)
        renderer.add_drawings(object_visuals)

        # steering_angle, speed, line_visuals = line_detection.process(frame)
        # renderer.add_drawings(line_visuals)

        renderer.render(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def get_connected_cameras(max_devices=5):
    conf = get_camera_config()
    print(conf)


if __name__ == '__main__':
    main('assets/v5_model.pt', 'video', 'assets/default.mp4')
