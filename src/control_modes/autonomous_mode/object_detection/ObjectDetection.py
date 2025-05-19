import os

import cv2
from .Detection import ObjectDetection
from .TrafficDetection import TrafficManager
from ....util.Render import Renderer

def main(weights_path: str, input_source: str, video_path: str = None):
    object_detector = ObjectDetection(weights_path, input_source)
    traffic_manager = TrafficManager()
    renderer = Renderer()
    
    if input_source == 'video' and video_path:
        cap = cv2.VideoCapture(video_path)
    else:
        cap = cv2.VideoCapture(0)  # Default to webcam
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        renderer.clear()

        traffic_state, detections, object_visuals = object_detector.process(frame)
        renderer.add_drawings(object_visuals)

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
