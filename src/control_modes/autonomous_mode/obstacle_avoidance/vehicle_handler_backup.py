from rplidar import RPLidar
from ..object_detection.Detection import ObjectDetection

import math
import numpy as np
import cv2
import torch
from math import floor

class VehicleHandler:
    def __init__(self, weights_path, input_source):
        self.lidar = RPLidar("Com3")
        self.object_detection = ObjectDetection(weights_path, input_source)
        self.cam = cv2.VideoCapture(1)
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path="assets/v5_model.pt", force_reload=True)
        self.focal_length = 540
        self.image_height = 1080
        self.image_width = 1920
        self.collision = False
        self.car_passed = True
        self.collision_stop = False

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

    def closest_distance(self, scan):
        scan_data = np.full(360, np.inf)

        for _, angle, distance in scan:
            angle_idx = min(359, floor(angle))
            scan_data[angle_idx] = distance

        closest_distance = np.min(scan_data)
        return closest_distance

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

    def detect_objects(self):
        while self.cam.isOpened():
            ret, frame = self.cam.read()
            if not ret:
                break
            detections = self.object_detection.detect_objects(frame)
            return detections

    def overlay_on_camera(self):
        for scan in self.iter_scans():
            ret, frame = self.cam.read()
            if not ret:
                break

            for angle, distance in scan:
                if distance < 100:
                    continue

                x, y = self.coordinate_conversion(angle, distance)
                u, v = self.homogeneous_coordinates(x, y)

                if 0 <= int(u) < self.image_width and 0 <= int(v) < self.image_height:
                    cv2.circle(frame, (int(u), int(v)), 3, (0, 255, 0), -1)

            cv2.imshow("LIDAR Projection", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cam.release()
        cv2.destroyAllWindows()

    def merge_measurements(self, detections, scan):
        object_distances = []

        for det in detections:
            x1, y1, x2, y2, conf, class_id = det[:6]
            distances = []
            class_name = self.model.names[int(class_id)]

            if class_name != "Car":
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
                    "class": class_name
                })

        return object_distances

    def start_overtake(self, object_distances):
        for obj in object_distances:
            if obj["distance"] < 3 and obj["class"] == "car":
                return True

    def check_collision(self, u, v, scan):
        min_angle = 160
        max_angle = 200
        for angle, distance in scan:
            if min_angle < max_angle:
                if min_angle <= angle <= max_angle:
                    if distance < 100:
                        continue
                    x, y = self.coordinate_conversion(angle, distance)
                    u_temp, v_temp = self.homogeneous_coordinates(x, y)
                    if 0 <= u_temp < self.image_width and 0 <= v_temp < self.image_height:
                        if 0 <= u - u_temp < 10 and 0 <= v - v_temp < 10:
                            self.collision = True
                            return self.collision

    def check_car(self, scan):
        min_angle = 260
        max_angle = 280
        for angle, distance in scan:
            if min_angle < max_angle:
                if min_angle <= angle <= max_angle:
                    if distance < 5000:
                        continue
                    self.car_passed = True
                    return self.car_passed

    def steer_left(self, object_distances):
        if self.start_overtake(object_distances):
            # steer left for overtaking
            print("starting overtake")

        if self.collision:
            # steer left to avoid car
            print("steering left to avoid car")
            self.collision = False

    def steer_right(self):
        if self.car_passed:
            # steer right to reemerge in original lane
            print("steering right to reemerge in original lane")

    def collision_stop(self, scan):
        min_angle = 160
        max_angle = 200
        for angle, distance in scan:
            if min_angle < max_angle:
                if min_angle <= angle <= max_angle:
                    if distance < 300:
                        self.collision_stop = True
                        return self.collision_stop

    def set_speed(self):
        if self.collision_stop:
            return 0

if __name__ == "__main__":
    weights_path = "assets/yolov5s.pt"
    input_source = "video"
    handler = VehicleHandler(weights_path, input_source)
    handler.overlay_on_camera()
