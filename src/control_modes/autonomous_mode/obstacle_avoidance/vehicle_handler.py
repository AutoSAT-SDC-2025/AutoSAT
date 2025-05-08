from .lidar_scan import LidarScans
from ..object_detection.Detection import ObjectDetection
from .rrttestright import RRTStar, Node
#from .....src.misc import calculate_steering, calculate_throttle

from rplidar import RPLidar
import cv2
import math
import torch
import numpy as np

class VehicleHandler:
    def __init__(self, weights_path, input_source):
        self.object_detection = ObjectDetection(weights_path, input_source)
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path="v5_model.pt", force_reload=True)
        self.cam = cv2.VideoCapture("assets/Car.mp4")
        self.lidar = RPLidar("COM3")
        self.waypoints = []
        self.initial_position = Node(0, 0)
        self.starting_position = Node(2, 0)
        self.lidar_scans = LidarScans()
        self.image_width = 1920
        self.image_height = 1080
        self.focal_length = 540
        self.car_detected = False
        self.collision = False

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
        x = (distance / 1000) * math.cos(radians)
        y = (distance / 1000) * math.sin(radians)
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

    def initialize_rrt(self, object_distances, goal):
        for obj in object_distances:
            if obj["class"] == "Car" and obj["distance"] < 5:
                self.car_detected = True
                start = self.starting_position
                rrt = RRTStar(start, goal, num_obstacles=1, map_size=[10,20])
                path = rrt.search()
                self.set_waypoints(path)
                return path

    def set_goal(self, object_distances):
        for obj in object_distances:
            if obj["distance"] < 3.0 and self.car_detected == True:
                    print("Nearby car detected")
                    distance_offset = 5.0
                    x_goal = obj["distance"] + distance_offset
                    y_goal = 0
                    goal = Node(x_goal, y_goal)
                    return goal

        print("No nearby car detected")
        return None

    def set_waypoints(self, waypoints):
        if waypoints:
            self.waypoints = waypoints
            print("Waypoints have been set:")
            for wp in waypoints:
                print(f"  -> x={wp[0]:.2f}, y={wp[1]:.2f}")
        else:
            print("No path found. Waypoints not set.")

    def steer_to_waypoint(self, current_pos):
        if not self.waypoints:
            print("No waypoints left.")
            return None

        wp = self.waypoints[0]
        dx = wp[0] - current_pos.x
        dy = wp[1] - current_pos.y

        if math.hypot(dx, dy) < 0.5:
            self.waypoints.pop(0)
            print("Waypoint reached, moving to next.")
            return self.steer_to_waypoint(current_pos)

        angle = math.atan2(dy, dx)
        print(f"Steering angle to next waypoint: {math.degrees(angle):.2f}°")
        return angle

    def set_steering(self, angle):
        if angle < 0:
            print("Steering left")

        elif angle > 0:
            print("Steering right")

        else:
            print("No steering required")
