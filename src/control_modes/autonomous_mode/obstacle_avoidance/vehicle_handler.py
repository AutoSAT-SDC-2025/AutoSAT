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

    def is_actual_waypoint(self, point, waypoints, threshold=0.05):
        """Check if the point exactly matches a predefined waypoint (allowing tiny float error)."""
        for wp in waypoints:
            if math.hypot(wp[0] - point[0], wp[1] - point[1]) < threshold:
                return True
        return False

    def is_too_close_to_predefined(self, point, waypoints, threshold=0.5):
        """Check if a point is within threshold of a waypoint — excluding exact match."""
        for wp in waypoints:
            if math.hypot(wp[0] - point[0], wp[1] - point[1]) < threshold:
                return True
        return False

    def set_waypoints(self):
        waypoints = [[-2.5, 3], [-2.5, 13], [-2, 15], [-1, 16], [-0.5, 17]]
        return waypoints

    def set_rrt(self, goal, detections, waypoints):
        if not self.path_found:
            start = [self.localizer.x, self.localizer.y]
            map_size = [20, 20]

            x = self.localizer.x
            y = self.localizer.y
            obstacles = self.set_obstacles(detections, x, y)

            path = []
            points = [start]

            # Always use the predefined waypoints
            points.extend(waypoints)

            # Filter out any passed-in waypoints that are too close to predefined ones
            if waypoints:
                for wp in waypoints:
                    if not self.is_too_close_to_predefined(wp, waypoints):
                        points.append(wp)

            # Add goal point
            points.append(goal[:2])

            for i in range(len(points) - 1):
                rrt_star = RRTStar(start=points[i], goal=points[i + 1], num_obstacles=0, map_size=map_size,
                                   obstacles=obstacles)
                segment_path = rrt_star.search()
                if not segment_path:
                    print(f"Failed to find path between {points[i]} and {points[i + 1]}")
                    return None

                segment_path = rrt_star.smooth_path(segment_path, iterations=200)
                # Remove duplicate point if path segments join exactly
                if path and segment_path[0] == path[-1]:
                    segment_path = segment_path[1:]

                path.extend(segment_path)

            self.path_found = True
            print("Full path with waypoints found:")
            for point in path:
                print(point)

            filtered_path = []
            for node in path:
                if self.is_actual_waypoint([node.x, node.y], waypoints) or not self.is_too_close_to_predefined([node.x, node.y], waypoints):
                    filtered_path.append(node)

            x_vals = [node.x for node in filtered_path]
            y_vals = [node.y for node in filtered_path]
            return x_vals, y_vals

    def set_goal(self, det, x, y, theta):
        if det["class"] == "Car" and self.goal_set is False:
            distance_offset = 12.0
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
                obstacles.append((-1.25 + x, det["distance"] + y, 2.5, 4))
        return obstacles

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

    def calculate_angle(self, x, y, waypoint):
        dx = waypoint[0] - x
        dy = waypoint[1] - y
        angle = math.atan2(dy,dx)
        return angle

    def angle_difference(self, desired_theta, current_theta):
        diff = desired_theta - current_theta
        angle_difference = (diff + math.pi) % (2 * math.pi) - math.pi  # Normalize to [-π, π]
        return angle_difference

    def steer_toward_waypoint(self, waypoint):
        current_x = self.localizer.x
        current_y = self.localizer.y
        current_theta = self.localizer.theta

        desired_theta = self.calculate_angle(current_x, current_y, waypoint)
        angle_diff = self.angle_difference(desired_theta, current_theta)

        if abs(angle_diff) < math.radians(5):
            print("Driving straight")
            self.can_controller.set_steering_and_throttle(0, 300)
        elif angle_diff > 0:
            print("Turning right")
            self.can_controller.set_steering_and_throttle(100, 300)
        else:
            print("Turning left")
            self.can_controller.set_steering_and_throttle(-100, 300)

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

                if det["class"] == "Car" and det["distance"] <= 8 and self.goal_set is False:
                    self.goal = self.set_goal(det, x, y, theta)
                    self.goal_set = True
                    if self.goal:
                        waypoints = self.set_waypoints()
                        result = self.set_rrt(self.goal, detections, waypoints)
                        if result:
                            self.x_vals, self.y_vals = result
                            self.path_found = True
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
                        self.steer_toward_waypoint((x_wp, y_wp))

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
            else:
                self.steer_toward_waypoint((x_wp, y_wp))


if __name__ == '__main__':
    weights_path = "assets/v5_model.pt"
    input_source = "video"

    localizer = Localizer()
    vehicle_handler = VehicleHandler(weights_path, input_source, localizer)
    vehicle_handler.main()