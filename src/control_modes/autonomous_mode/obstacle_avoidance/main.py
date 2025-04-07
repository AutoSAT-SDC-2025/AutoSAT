from src.obstacle_avoidance import pedestrian_handler, vehicle_handler
from src.object_detection.detection import ObjectDetection
import cv2

def main():
    cam = cv2.VideoCapture(1)
    detector = ObjectDetection()

    ped_handler = pedestrian_handler()
    veh_handler = vehicle_handler()
    
    while cam.isOpened():
        ret, frame = cam.read()
        if not ret:
            break

        detections = detector.detect(frame)

        ped_result = ped_handler.main(detections)

        vehicle_data = veh_handler.process_vehicle(detections)
        overtake_state = veh_handler.overtake_vehicle(vehicle_data['Vehicle seen'])

        print("Pedestrian Safe to Pass:", ped_result)
        print("Vehicle State:", overtake_state)

        cv2.imshow("Camera", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()