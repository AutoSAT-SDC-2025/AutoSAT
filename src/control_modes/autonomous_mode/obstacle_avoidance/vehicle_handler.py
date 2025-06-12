from .RRT import RRTStar
from src.util.video import get_camera_config
from ..localization.localization import Localizer
from ..object_detection.Detection import ObjectDetection
from ..line_detection.LineDetection import LineFollowingNavigation
from ....car_variables import KartGearBox
from rplidar import RPLidar
from math import floor
# import cv2
import math
import matplotlib.pyplot as plt
import numpy as np
import time


class VehicleHandler:
    def __init__(self, weights_path=None, input_source=None, localizer=None, can_controller=None, car_type=None):
        self.localizer = localizer
        self.can_controller = can_controller
        self.car_type = car_type
        self.cams = get_camera_config()
        self.object_detection = ObjectDetection(weights_path, input_source)
        self.lane_navigator = LineFollowingNavigation()
        self.lidar = RPLidar("/dev/ttyUSB0")
        self.car_detected = False
        self.collision = False
        self.goal_set = False
        self.path_found = False
        self.goal = None
        self.center_start_timer = None
        self.centered = False
        self.steering_state = None
        self.start_timer = None
        self.scan_started = False
        self.car_passed = False
        self.overtake_completed = False

    def is_actual_waypoint(self, point, waypoints, threshold=0.05):
        for wp in waypoints:
            if math.hypot(wp[0] - point[0], wp[1] - point[1]) < threshold:
                return True
        return False

    def is_too_close_to_predefined(self, point, waypoints, threshold=0.5):
        for wp in waypoints:
            if math.hypot(wp[0] - point[0], wp[1] - point[1]) < threshold:
                return True
        return False

    def set_waypoints(self, x, y):
        waypoints = [[-0.25 + x, 0.75 + y], [-0.5 + x, 1 + y],
                     [-0.75 + x, 1.25 + y], [-1 + x, 1.5 + y],
                     [-1.5 + x, 1.75 + y], [-2 + x, 2 + y],
                     [-2.25 + x, 2.5 + y], [-2.5 + x, 3 + y],
                     [-2.5 + x, 14 + y], [-2.25 + x, 15 + y],
                     [-2 + x, 16 + y], [-1.5 + x, 16.5 + y],
                     [-1 + x, 17 + y], [-0.75 + x, 17.5 + y],
                     [-0.5 + x, 18 + y]]
        return waypoints

    def set_rrt(self, goal, detections, waypoints):
        if not self.path_found:
            x = self.localizer.x / 1000
            y = self.localizer.y / 1000

            start = [x, y]
            map_size = [20, 20]

            obstacles = self.set_obstacles(detections, x, y)

            path = []
            points = [start]

            points.extend(waypoints)

            if waypoints:
                for wp in waypoints:
                    if not self.is_too_close_to_predefined(wp, waypoints):
                        points.append(wp)

            points.append(goal[:2])

            for i in range(len(points) - 1):
                rrt_star = RRTStar(start=points[i], goal=points[i + 1], num_obstacles=0, map_size=map_size,
                                   obstacles=obstacles)
                segment_path = rrt_star.search()
                if not segment_path:
                    print(f"Failed to find path between {points[i]} and {points[i + 1]}")
                    return None

                segment_path = rrt_star.smooth_path(segment_path, iterations=200)
                if path and segment_path[0] == path[-1]:
                    segment_path = segment_path[1:]

                path.extend(segment_path)

            self.path_found = True
            print("Full path with waypoints found:")
            for point in path:
                print(point)

            filtered_path = []
            for node in path:
                if self.is_actual_waypoint([node.x, node.y], waypoints) or not self.is_too_close_to_predefined(
                        [node.x, node.y], waypoints):
                    filtered_path.append(node)

            x_vals = [node.x if hasattr(node, "x") else node[0] for node in filtered_path]
            y_vals = [node.y if hasattr(node, "y") else node[1] for node in filtered_path]
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
        angle = round(math.atan2(dy, dx))
        return angle

    def angle_difference(self, desired_theta, current_theta):
        diff = desired_theta - current_theta
        return -((diff + math.pi) % (2 * math.pi) - math.pi)

    def adjust_steering(self, steering_angle):
        if self.car_type == 'Hunter':
            new_steering_angle = round(steering_angle * 576 / 90)

            return max(round(min(new_steering_angle, 576), -576))
        else:
            new_steering_angle = round(steering_angle * (1.25 / 30))
            return max(min(new_steering_angle, 1.25), -1.25)

    def set_steering_angle(self, angle_difference, steering_angle=None):
        angle_in_degrees = math.degrees(angle_difference)
        new_steering_angle = self.adjust_steering(angle_in_degrees)
        print(f"Steering angle diff (deg): {angle_in_degrees:.2f}, Command: {new_steering_angle:.2f}")
        if self.car_type == 'Hunter':
            self.can_controller.set_steering_and_throttle(new_steering_angle, 300)
        else:
            self.can_controller.set_steering(steering_angle)

    def steer_toward_waypoint(self, waypoint):
        current_x = self.localizer.x / 1000
        current_y = self.localizer.y / 1000
        current_theta = self.localizer.theta

        desired_theta = self.calculate_angle(current_x, current_y, waypoint)
        angle_diff = self.angle_difference(desired_theta, current_theta)
        self.set_steering_angle(angle_diff)

    def plot_waypoints(self, goal, detections, x_vals, y_vals):
        x = self.localizer.x / 1000
        y = self.localizer.y / 1000
        obstacles = self.set_obstacles(detections, x, y)
        plt.figure(figsize=(8, 8))
        plt.plot(x_vals, y_vals, marker='o', color='blue', label='RRT* Path')
        plt.scatter([x_vals[0]], [y_vals[0]], color='green', label='Start')
        plt.scatter([goal[0]], [goal[1]], color='red', label='Goal')

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

    def determine_closest(self, scan):
        """Find and print the closest object distance from the scan data."""
        scan_data = np.full(360, np.inf)

        for _, angle, distance in scan:
            angle_idx = min(359, floor(angle))

            if distance < 150:  # Ignore very close objects
                scan_data[angle_idx] = np.inf
            else:
                scan_data[angle_idx] = distance

        closest_distance = np.min(scan_data)
        print(f"Closest object distance: {closest_distance:.2f} mm")
        return closest_distance

    def check_collision(self, closest_distance):
        if closest_distance < 500:
            print("Collision detected, stopping the car")
            if self.car_type == 'Hunter':
                self.can_controller.set_steering_and_throttle(0, 0)
            else:
                self.can_controller.set_break(100)
            self.collision = True
            return True
        return False

    def main(self, front_view=None):

        _, detections, _ = self.object_detection.process(front_view)

        x = self.localizer.x / 1000
        y = self.localizer.y / 1000
        theta = self.localizer.theta

        for det in detections:
            if det["class"] == "Car" and det["distance"] <= 10 and not self.goal_set:
                self.goal = self.set_goal(det, x, y, theta)
                self.goal_set = True

                if self.goal:
                    waypoints =  self.set_waypoints(x, y)
                    result = self.set_rrt(self.goal, detections, waypoints)
                    if result:
                        self.x_vals, self.y_vals = result
                        self.path_found = True
                        self.detections = detections

        if not self.collision:
            if self.goal_set and self.path_found and self.x_vals and self.y_vals:
                x_wp, y_wp = self.x_vals[0], self.y_vals[0]

                if self.waypoint_reached(x_wp, y_wp):
                    print("Waypoint reached.")
                    self.x_vals.pop(0)
                    self.y_vals.pop(0)

                self.steer_toward_waypoint((x_wp, y_wp))

        if self.goal_set and self.path_found and not self.x_vals:
            if self.goal_reached():
                print("Final goal reached!")
                if self.car_type == 'Hunter':
                    self.can_controller.set_steering_and_throttle(0, 0)
                else:
                    self.can_controller.set_steering(0)
                    self.can_controller.set_throttle(50)

    def steer_to_centre(self, detections=None, center_x=None):
        """if not detections is not None:
            print("No detections available for steering.")
            return False

        # Get lane information
        lane_width = 3  # meters
        scaling_factor = lane_width / 2
        lane_offset = (center_x - (CameraResolution.WIDTH / 2)) / CameraResolution.WIDTH * scaling_factor

        # Get vehicle offset from detected object
        vehicle_offset = 0
        for det in detections:
            if det['class'] == "Car":
                x1, y1, x2, y2 = det['bbox']
                object_center_x = (x1 + x2) / 2
                screen_center_x = CameraResolution.WIDTH / 2
                vehicle_offset = object_center_x - screen_center_x

        # Combine both offsets with weights
        lane_weight = 0.7  # Prioritize lane centering
        vehicle_weight = 0.3  # Secondary priority to vehicle avoidance
        combined_offset = (lane_weight * lane_offset + vehicle_weight * vehicle_offset)

        # Calculate steering angle
        Kp = 0.5
        steering_angle = round(Kp * combined_offset)
        if self.car_type == 'Hunter':
            MAX_STEERING_ANGLE = 576  # Maximum steering angle in CAN units for Hunter
            steering_angle = max(round(min(steering_angle, MAX_STEERING_ANGLE), -MAX_STEERING_ANGLE))
        else:
            MAX_STEERING_ANGLE = 1.25  # Maximum steering angle for Kart
            steering_angle = max(min(steering_angle, MAX_STEERING_ANGLE), -MAX_STEERING_ANGLE)
        # Check if we're centered
        DEADZONE = 0.1 if self.car_type != 'Hunter' else 10  # Adjust deadzone based on car type
        if abs(steering_angle) < DEADZONE:
            print("Vehicle centered in lane. Stopping steering adjustments.")
            steering_angle = 0
            self.centered = True
        else:
            self.centered = False
            print(f"Steering {'right' if combined_offset > 0 else 'left'} by {steering_angle} CAN units.")
            print(f"Lane offset: {lane_offset:.2f}m, Vehicle offset: {vehicle_offset:.2f}px")

        # Apply steering
        if self.car_type == 'Hunter':
            self.can_controller.set_steering_and_throttle(steering_angle, 300)
        else:
            self.can_controller.set_steering(steering_angle)
            self.can_controller.set_throttle(50)"""
        self.centered = True
        return self.centered

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

    def get_scan(self):
        scan_data = []
        for _, angle, distance in self.lidar.iter_measures():
            if len(scan_data) > 5:
                return scan_data
            scan_data.append((angle, distance))

    def check_collision_during_overtake(self):
        try:
            scan = self.get_scan()
            if scan:
                closest_distance = self.determine_closest(scan)
                if self.check_collision(closest_distance):
                    print("Emergency stop - collision detected during overtake!")
                    return True
            return False
        except Exception as e:
            print(f"Error checking collisions: {e}")
            return False

    def scan_right(self):
        for scan in self.iter_scans():
            obstacle_found = False
            for _, angle, distance in scan:
                if angle == 270 and distance < 4000:
                    obstacle_found = True
                    print("Obstacle detected on the right. Continuing scan...")
                    break

            if not obstacle_found:
                print("No obstacle on the right. Stopping scan.")
                self.car_passed = True
                break

            time.sleep(0.1)

    def manual_main(self, front_view=None):
        _, detections, _ = self.object_detection.process(front_view)
        print(f"Detections type: {type(detections)}")
        print(f"Detections content: {detections}")

        current_time = time.time()

        if not self.steering_state:
            # center_x = self.lane_navigator.get_center_x(front_view)
            # self.centered = self.steer_to_centre(detections, center_x)
            if self.centered is True:
                if not hasattr(self, 'center_start_timer'):
                    self.center_start_timer = current_time
                elif (current_time - self.center_start_timer) >= 0.5:
                    print("Vehicle centered. Starting steering sequence.")
                    self.steering_state = 'Left'
                    self.start_timer = current_time
            else:
                self.center_start_timer = None
            return
        if self.centered is True:
            # Steer to the left
            if self.car_type == 'Hunter':
                self.can_controller.set_steering_and_throttle(-100, 300)
            else:
                self.can_controller.set_kart_gearbox(KartGearBox.forward)
                self.can_controller.set_throttle(50)
                self.can_controller.set_steering(-1.25)  # max steering is between -1.25 and 1.25
            if self.check_collision_during_overtake():
                self.steering_state = None
                return
            self.steering_state = 'Left'
            self.start_timer = current_time

            if self.steering_state == 'Left' and (current_time - self.start_timer) >= 1:
                # Steer to the right
                if self.car_type == 'Hunter':
                    self.can_controller.set_steering_and_throttle(100, 300)
                else:
                    self.can_controller.set_kart_gearbox(KartGearBox.forward)
                    self.can_controller.set_throttle(50)
                    self.can_controller.set_steering(1.25)
                if self.check_collision_during_overtake():
                    self.steering_state = None
                    return
                self.steering_state = 'Right'
                self.start_timer = time.time()

            elif self.steering_state == 'Right' and (current_time - self.start_timer) >= 1:
                if self.car_type == 'Hunter':
                    self.can_controller.set_steering_and_throttle(0, 300)
                else:
                    self.can_controller.set_kart_gearbox(KartGearBox.forward)
                    self.can_controller.set_throttle(100)
                    self.can_controller.set_steering(0)
                if self.check_collision_during_overtake():
                    self.steering_state = None
                    return
                self.steering_state = 'Switched'
                self.start_timer = time.time()
                self.scan_started = False

            elif self.steering_state == 'Switched':
                self.can_controller.set_steering_and_throttle(0, 300)

                if (current_time - self.start_timer) >= 3 and not self.scan_started:
                    self.scan_started = True
                    self.scan_right()
                if self.car_passed is True:
                    print("Car passed. Continuing steering sequence.")
                    if self.car_type == 'Hunter':
                        self.can_controller.set_steering_and_throttle(100, 300)
                    else:
                        self.can_controller.set_kart_gearbox(KartGearBox.forward)
                        self.can_controller.set_throttle(50)
                        self.can_controller.set_steering(1.25)
                    if self.check_collision_during_overtake():
                        self.steering_state = None
                        return
                    self.steering_state = 'RightReturn'
                    self.start_timer = time.time()
                else:
                    print("Waiting for car to pass...")

            elif self.steering_state == 'RightReturn' and self.car_passed is True and (
                    current_time - self.start_timer) >= 1:
                if self.car_type == 'Hunter':
                    self.can_controller.set_steering_and_throttle(-100, 300)
                else:
                    self.can_controller.set_kart_gearbox(KartGearBox.forward)
                    self.can_controller.set_throttle(50)
                    self.can_controller.set_steering(-1.25)
                if self.check_collision_during_overtake():
                    self.steering_state = None
                    return
                self.steering_state = 'LeftReturn'
                self.start_timer = time.time()

            elif self.steering_state == 'LeftReturn' and self.car_passed is True and (
                    current_time - self.start_timer) >= 1:
                if self.car_type == 'Hunter':
                    self.can_controller.set_steering_and_throttle(0, 300)
                else:
                    self.can_controller.set_kart_gearbox(KartGearBox.forward)
                    self.can_controller.set_throttle(100)
                    self.can_controller.set_steering(0)
                if self.check_collision_during_overtake():
                    self.steering_state = None
                    return
                self.steering_state = 'Done'
                self.overtake_completed = True


if __name__ == '__main__':
    weights_path = "assets/v5_model.pt"
    input_source = "video"

    localizer = Localizer()
    vehicle_handler = VehicleHandler(weights_path, input_source, localizer)
    vehicle_handler.manual_main()