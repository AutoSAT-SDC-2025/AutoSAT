from rplidar import RPLidar
import cv2
import torch
import math
import numpy as np
from ..object_detection.Detection import ObjectDetection
from ....can_interface.bus_connection import connect_to_can_interface, disconnect_from_can_interface
from ....can_interface.can_factory import select_can_controller_creator, create_can_controller
from src.util.video import get_camera_config

class PedestrianHandler:
    def __init__(self, weights_path, input_source):
        self.captures = None
        self.car_type = "Hunter"
        self.can_bus = connect_to_can_interface(0)

        self.can_creator = select_can_controller_creator(self.car_type)
        self.can_controller = create_can_controller(self.can_creator, self.can_bus)
        self.cams = get_camera_config()
        self.lidar = RPLidar("Com3")
        self.object_detection = ObjectDetection(weights_path, input_source)
        self.cam = cv2.VideoCapture(1)
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path="assets/v5_model.pt", force_reload=True)
        self.car_distance_threshold = 2
        self.previous_detection = {}
        self.focal_length = 540
        self.image_height = 1080
        self.image_width = 1920
        self.initial_position = None
        self.current_position = None
        self.direction = None


    def detect_objects(self):
        while self.cam.isOpened():
            ret, frame = self.cam.read()
            if not ret:
                break
            detections = self.object_detection.detect_objects(frame)
            return detections

    def get_initial_position(self, detections):
        for det in detections:
            if det["class"] == "Person":
                x1, _, x2, _ = det["bbox"]
                if (x1 + x2) / 2 < self.image_width / 2:
                    self.initial_position = "Left"
                    #return self.initial_position
                else:
                    self.initial_position = "Right"
                    #return self.initial_position

    def get_direction(self, detections):
        for det in detections:
            if det["class"] == "Person":
                x1, y1, x2, y2 = det["bbox"]
                x_center = (x1 + x2) / 2
                obj_id = "Person"

                if obj_id in self.previous_detection:
                    prev_x_center = self.previous_detection[obj_id]["x_center"]

                    if x_center > prev_x_center + 5:
                        self.direction = "Right"
                    elif x_center < prev_x_center - 5:
                        self.direction = "Left"
                    else:
                        self.direction = "Stationary"
                else:
                    self.direction = "Unknown"

                self.previous_detection[obj_id] = {"x_center": x_center}

        return self.direction

    def get_current_pos(self, detections):
        for det in detections:
            if det["class"] == "Person":
                x_center = (det["bbox"][0] + det["bbox"][2]) / 2

                if x_center < self.image_width / 2:
                    self.current_position = "Left"
                elif x_center > self.image_width / 2:
                    self.current_position = "Right"
                else:
                    self.current_position = "Unknown"
        return self.current_position

    def pedestrian_crossed(self):
        if self.current_position == "Left" and self.initial_position == "Right" and self.direction == "Stationary":
            return True
        elif self.current_position == "Right" and self.initial_position == "Left" and self.direction == "Stationary":
            return True
        return False

    def iter_scans(self):
        lidar_scan = []
        try:
            for new_scan, _, angle, distance in self.lidar.iter_measures():
                if new_scan:
                    if len(lidar_scan) > 5:
                        yield lidar_scan
                    lidar_scan = []

                lidar_scan.append((angle, distance))

        except KeyboardInterrupt:
            print("Stopping Lidar")
        finally:
            self.lidar.stop()
            self.lidar.disconnect()

    def coordinate_conversion(self, angle, distance):
        radians = math.radians(angle)
        x = (distance / 1000.0) * math.cos(radians)
        y = (distance / 1000.0) * math.sin(radians)
        return x, y

    def homogeneous_coordinates(self, x, y):
        k = np.array([
            [self.focal_length, 0, self.image_width / 2],
            [0, self.focal_length, self.image_height / 2],
            [0, 0, 1]
        ])

        t = np.array([[0], [0.3], [0.5]])
        R = np.array([[0, -1, 0], [-1, 0, 0], [0, 0, 1]])

        extrinsics = np.hstack((R, t))

        homogeneous_point = np.array([x, y, 0, 1])
        projection_matrix = np.dot(k, extrinsics)
        projected_2d_homogeneous = np.dot(projection_matrix, homogeneous_point)

        u = projected_2d_homogeneous[0] / projected_2d_homogeneous[2]
        v = projected_2d_homogeneous[1] / projected_2d_homogeneous[2]
        return u, v

    def determine_distance(self, detections, scan):
        object_distances = []

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            distances = []

            if det["class"] != "Person":
                continue

            for angle, distance in scan:
                if distance < 100:
                    continue
                x, y = self.coordinate_conversion(angle, distance)
                u, v = self.homogeneous_coordinates(x, y)

                if x1 <= u <= x2 and y1 <= v <= y2:
                    distances.append(distance / 1000.0)

            if distances:
                object_distances.append({
                    "box": (x1, y1, x2, y2),
                    "distance": np.mean(distances),
                    "class": "Person"
                })

        return object_distances

    def stop_car(self, object_distances):
        for obj in object_distances:
            if obj["distance"] < self.car_distance_threshold and obj["class"] == "Person":
                self.can_controller.set_steering_and_throttle(0, 0)
                self.can_controller.set_parking_mode(1)
            else:
                self.can_controller.set_parking_mode(0)
                self.can_controller.set_steering_and_throttle(0, 300)

    def continue_driving(self):
        if self.pedestrian_crossed():
            self.can_controller.set_parking_mode(0)
            self.can_controller.set_steering_and_throttle(0, 300)

def main():
    weights_path = "assets/v5_model.pt"
    input_source = "video"
    handler = PedestrianHandler(weights_path, input_source)

    initial_position_set = False

    try:
        for scan in handler.iter_scans():
            ret, frame = handler.cam.read()
            if not ret:
                print("Failed to read from camera.")
                break

            # Run object detection
            detections = handler.object_detection.detect_objects(frame)

            if not detections:
                continue

            # Get pedestrian initial position once
            if not initial_position_set:
                handler.get_initial_position(detections)
                initial_position_set = True

            # Update position and direction
            handler.get_current_pos(detections)
            handler.get_direction(detections)

            # Determine distance to pedestrians
            object_distances = handler.determine_distance(detections, scan)

            # Decide to stop or continue
            handler.stop_car(object_distances)
            handler.continue_driving()

    except KeyboardInterrupt:
        print("Interrupted by user. Stopping.")
    finally:
        handler.lidar.stop()
        handler.lidar.disconnect()
        handler.cam.release()
        disconnect_from_can_interface(handler.can_bus)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

