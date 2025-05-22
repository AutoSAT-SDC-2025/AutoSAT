from .RRT import RRTStar
from ....can_interface.bus_connection import connect_to_can_interface
from ....can_interface.can_factory import select_can_controller_creator, create_can_controller
from src.util.video import get_camera_config
from ..localization.localization import Localizer
from .CameraClassification import ScanMerging

from rplidar import RPLidar
import cv2
import math
import matplotlib.pyplot as plt

class VehicleHandler:
    def __init__(self, weights_path, input_source, localizer):
        self.localizer = localizer
        self.captures = None
        self.car_type = "Hunter"
        self.can_bus = connect_to_can_interface(0)

        self.can_creator = select_can_controller_creator(self.car_type)
        self.can_controller = create_can_controller(self.can_creator, self.can_bus)
        self.cams = get_camera_config()
        self.scan_merging = ScanMerging(weights_path, input_source)
        self.cam = cv2.VideoCapture(1)
        self.lidar = RPLidar("COM3")
        self.image_width = 1920
        self.image_height = 1080
        self.focal_length = 540
        self.car_detected = False
        self.collision = False
        self.goal_set = False
        self.path_found = False

    """def check_collision(self, u, v, scan):
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
                            return self.collision"""


    def set_rrt(self, goal, detections):
        if self.path_found is False:
            start = [self.localizer.x, self.localizer.y]
            map_size = [20, 20]

            x = self.localizer.x
            y = self.localizer.y
            obstacles = self.set_obstacles(detections, x, y)

            rrt_star = RRTStar(start=start, goal=goal, num_obstacles=0, map_size=map_size, obstacles=obstacles)
            path = rrt_star.search()

            if path:
                smoothed_path = rrt_star.smooth_path(path, iterations=200)
                path = smoothed_path
                rrt_star.path = path  # in case draw_path() relies on internal path
                rrt_star.draw_path()

                self.path_found = True
                print("RRT* path found:")
                for point in path:
                    print(point)

                x_vals = [node.x for node in path]
                y_vals = [node.y for node in path]

                return x_vals, y_vals

            else:
                print("RRT* failed to find a path.")

    def set_goal(self, det, x, y, theta):
        if det["class"] == "Car" and self.goal_set is False:
            distance_offset = 8.0
            x_goal = x
            y_goal = det["distance"] + distance_offset + y
            theta = theta
            goal = (x_goal, y_goal, theta)
            print(f"Goal set to: {goal} at distance {det['distance']:.2f}m")
            return goal
        elif det["class"] == "Car" and self.goal_set is True:
            print("Goal is already set.")
        else:
            print("No suitable car detected to set goal.")
            return None

    def set_obstacles(self, detections, x, y):
        obstacles = []
        for det in detections:
            if det["class"] == "Car":
                ox = -1.25 + x
                oy = det["distance"] + y
                width = 2.5
                height = 4
                obstacles.append((ox, oy, width, height))
        return obstacles

    def angle_to_waypoint(self, x, y, path):
        for point in path:
            dx = point[0] - x
            dy = point[1] - y

            angle = math.atan2(dy, dx)
            print(f"Steering angle to next waypoint: {math.degrees(angle):.2f}°")
            if angle == 0:
                print("No steering angle required")
            return angle

    def waypoint_reached(self, x_wp, y_wp, threshold=0.1):
        current_x, current_y = self.localizer.x, self.localizer.y
        dx = x_wp - current_x
        dy = y_wp - current_y
        if math.hypot(dx, dy) < threshold:
            return True

    def goal_reached(self, threshold=0.1):
        if not self.goal:
            return False
        goal_x, goal_y, goal_theta = self.goal
        return self.waypoint_reached(goal_x, goal_y, threshold)

    def set_steering(self, angle):
        if angle < 0:
            print("Steering left")
            self.can_controller.set_steering_and_throttle(-100, 300)
        elif angle > 0:
            print("Steering right")
            self.can_controller.set_steering_and_throttle(100, 300)
        else:
            print("No steering required")
            self.can_controller.set_steering_and_throttle(0, 300)

    def plot_waypoints(self, goal, detections, x_vals, y_vals):
        x = self.localizer.x
        y = self.localizer.y
        obstacles = self.set_obstacles(detections, x, y)
        plt.figure(figsize=(8, 8))
        plt.plot(x_vals, y_vals, marker='o', color='blue', label='RRT* Path')
        plt.scatter([x_vals[0]], [y_vals[0]], color='green', label='Start')
        plt.scatter([goal[0]], [goal[1]], color='red', label='Goal')

        # Plot obstacles (optional, if you want to visualize them)
        for ox, oy, width, height in obstacles:
            rect = plt.Rectangle((ox, oy), width, height, color='gray', alpha=0.5)
            plt.gca().add_patch(rect)

        plt.xlim(-5, 5)
        plt.ylim(0, 15)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('RRT* Path Planning')
        plt.legend()
        plt.grid(True)
        plt.show()

    def main(self):
        while self.cam.isOpened():
            ret, frame = self.cam.read()
            if not ret:
                break

            frame, detections = self.scan_merging.get_frame_with_detections()

            x = self.localizer.x
            y = self.localizer.y
            theta = self.localizer.theta

            for det in detections:
                x1, y1, x2, y2 = det['bbox']
                label = f"{det['class']} ({det['distance']:.2f}m)"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                if det["class"] == "Car" and det["distance"] <= 5 and self.goal_set is False:
                    self.goal = self.set_goal(det, x, y, theta)  # Pass x, y to set_goal
                    self.goal_set = True
                    if self.goal:
                        result = self.set_rrt(self.goal, detections)
                        if result:
                            self.x_vals, self.y_vals = result
                            self.detections = detections

            cv2.imshow('Detection', frame)

            if self.goal_set and self.path_found:
                if self.x_vals and self.y_vals:
                    x_wp, y_wp = self.x_vals[0], self.y_vals[0]
                    if self.waypoint_reached(x_wp, y_wp):
                        print("Waypoint reached.")
                        self.x_vals.pop(0)
                        self.y_vals.pop(0)
                        continue
                    elif len(self.x_vals) > 0:
                        angle = self.angle_to_waypoint(self.localizer.x, self.localizer.y, [(x_wp, y_wp)])
                        self.set_steering(angle)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            if self.goal_set and self.path_found and not self.x_vals:
                if self.goal_reached():
                    print("Final goal reached!")
                    break

        self.cam.release()
        cv2.destroyAllWindows()

        if self.goal_set and self.path_found and self.x_vals and self.y_vals:
            x_wp, y_wp = self.x_vals[0], self.y_vals[0]

            if self.waypoint_reached(x_wp, y_wp):
                print("Waypoint reached.")
                self.x_vals.pop(0)
                self.y_vals.pop(0)

            if self.x_vals:
                angle = self.angle_to_waypoint(self.localizer.x, self.localizer.y, [(x_wp, y_wp)])
                self.set_steering(angle)
            elif self.goal_reached():
                print("Final goal reached!")
                self.set_steering(0)

if __name__ == '__main__':
    weights_path = "assets/v5_model.pt"
    input_source = "video"

    localizer = Localizer()
    vehicle_handler = VehicleHandler(weights_path, input_source, localizer)
    vehicle_handler.main()