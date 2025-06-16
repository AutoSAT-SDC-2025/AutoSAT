python
import os

import cv2
from .Detection import ObjectDetection
from .TrafficDetection import TrafficManager
from ....util.Render import Renderer


def main(weights_path: str, input_source: str, video_path: str = None):
    """
    Entry point for running object and traffic detection.
    Initializes the detector, traffic manager, and renderer.
    Handles video or webcam input, processes each frame, and visualizes results.
    """
    object_detector = ObjectDetection(weights_path, input_source)
    traffic_manager = TrafficManager()
    renderer = Renderer()

    # Select video source: file or webcam
    if input_source == 'video' and video_path:
        cap = cv2.VideoCapture(video_path)
    else:
        cap = cv2.VideoCapture(0)  # Default to webcam

    # Main loop: process each frame
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        renderer.clear()

        # Run detection and get drawing instructions
        traffic_state, detections, object_visuals = object_detector.process(frame)
        renderer.add_drawings(object_visuals)

        # Draw results and display
        renderer.draw(frame)
        cv2.imshow('Traffic Detection', frame)
        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Print current working directory for debugging
    # the __main__ is only ran when this file is executed directly
    print("Current working dir:", os.getcwd())

    # Run detection using the YOLOv5 model on a default video
    main('assets/v5_model.pt', 'video', 'assets/default.mp4')