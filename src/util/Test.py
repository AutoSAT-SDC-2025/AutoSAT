import os

import cv2
from ..control_modes.autonomous_mode.object_detection.ObjectDetection import ObjectDetection
from ..control_modes.autonomous_mode.object_detection.TrafficDetection import TrafficManager

from ..control_modes.autonomous_mode.line_detection.LineDetection import LineFollowingNavigation

from .Render import Renderer

def main(weights_path: str, input_source: str, video_path: str = None):
    object_detector = ObjectDetection(weights_path, input_source)
    traffic_manager = TrafficManager()
    nav = LineFollowingNavigation(width=848, height=480)
    renderer = Renderer()
    

    cap = cv2.VideoCapture(video_path)

    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        renderer.clear()

        traffic_state, detections, object_visuals = object_detector.process(frame)
        renderer.add_drawings(object_visuals)

        steering_angle, speed, line_visuals = nav.process(frame)
        renderer.add_drawings(line_visuals)

        renderer.draw(frame)
        cv2.imshow('Traffic Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("Current working dir:", os.getcwd())

    # Use v5_model.pt (model from last year) on the video (default.mp4)
    main('assets/v5_model.pt', 'video', 'assets/default.mp4')
