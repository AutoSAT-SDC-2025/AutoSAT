from ..object_detection.Detection import ObjectDetection
from .pedestrian_handler import PedestrianHandler
import cv2

def main():
    pedestrian_handler = PedestrianHandler(weights_path="assets/v5_model.pt", input_source="video")

    cap = cv2.VideoCapture("assets/Person.mp4")
    if not cap.isOpened():
        print("Cannot open video file.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Cannot read frame")
            break

        detections = pedestrian_handler.detect_objects(frame)
        directions = pedestrian_handler.get_direction(detections)
        current_position = pedestrian_handler.get_current_pos(detections)
        safe_positions = pedestrian_handler.get_safe_pos(detections, directions)

        print("Detections:", detections)
        print("Directions:", directions)
        print("Current position:", current_position)
        print("Safe positions:", safe_positions)

        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            class_label = det['class']
            distance = det['distance']

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            text = f"{class_label} ({distance:.2f}m)"
            cv2.putText(frame, text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Frame Preview", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
