import cv2
#from src.control_modes.autonomous_mode.object_detection.TrafficDetection import TrafficManager
from src.control_modes.autonomous_mode.object_detection.Detection import ObjectDetection
from src.pedestrian_handler import PedestrianHandler

def main(weights_path: str, input_source: str, video_path: str = None):
    pedestrian_handler = PedestrianHandler()
    object_detector = ObjectDetection(weights_path, input_source)

    if input_source == 'video' and video_path:
        cam = cv2.VideoCapture(video_path)
    else:
        cam = cv2.VideoCapture(0)  # Default to webcam

    if not cam.isOpened():
        print("Could not open camera or video.")
        return

    while cam.isOpened():
        ret, frame = cam.read()
        if not ret or frame is None:
            print("Failed to read frame or video has ended.")
            break

        detections = object_detector.detect_objects(frame)

        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            class_label = det['class']
            distance = det['distance']
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{class_label} ({distance:.2f}m)", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        pedestrian_safe = pedestrian_handler.main(detections)
        print("Pedestrian Safe:", pedestrian_safe)

        cv2.imshow('Traffic Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main('assets/v5_model.pt', 'video', 'assets/Person.mp4')
