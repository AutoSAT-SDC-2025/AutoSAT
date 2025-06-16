from .RRT import RRTStar
from src.util.video import get_camera_config
from ..localization.localization import Localizer
from ..object_detection.Detection import ObjectDetection
from ..line_detection.LineDetection import LineFollowingNavigation
from ....car_variables import KartGearBox, CameraResolution
from rplidar import RPLidar
from math import floor
import math
import matplotlib.pyplot as plt
import numpy as np
import time

class VehicleHandler:
    """
    Handles the vehicle’s perception, planning, and basic control logic.

    Attributes
    ----------
    localizer : Localizer
        Provides the current position and orientation of the vehicle.
    can_controller : object
        Interface to send control commands (steering, throttle) to the vehicle.
    car_type : str
        Type of vehicle ('Hunter' or other), used for adjusting control parameters.
    cams : dict
        Dictionary containing camera configurations.
    object_detection : ObjectDetection
        Handles detection of objects such as other cars in the environment.
    lane_navigator : LineFollowingNavigation
        Module for handling lane following behavior.
    lidar : RPLidar
        Lidar sensor for detecting nearby obstacles.
    car_detected : bool
        Flag indicating whether a car has been detected in front.
    collision : bool
        Indicates whether a collision is imminent or has occurred.
    goal_set : bool
        True if a goal has been defined based on object detection.
    path_found : bool
        True if a full RRT path to the goal has been generated.
    goal : tuple
        Target goal in the format (x, y, theta).
    center_start_timer : any
        Timer to handle centering logic (if used).
    centered : bool
        True if the vehicle is aligned in its lane or path.
    steering_state : any
        Used to maintain or switch steering states.
    start_timer : any
        Timer used to delay behavior (e.g., starting line following).
    scan_started : bool
        True if LIDAR scan has started.
    car_passed : bool
        True if the vehicle has passed another car (e.g., overtaking).
    overtake_completed : bool
        Indicates whether overtaking maneuver is completed.
    """

    def __init__(self, weights_path=None, input_source=None, localizer=None, can_controller=None, car_type=None):
        # Initialize core modules
        self.localizer = localizer
        self.can_controller = can_controller
        self.car_type = car_type

        # Initialize subsystems
        self.cams = get_camera_config()
        self.object_detection = ObjectDetection(weights_path, input_source)
        self.lane_navigator = LineFollowingNavigation()
        self.lidar = RPLidar("/dev/ttyUSB0")

        # Control and state flags
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
        """Check if the point exactly matches a defined waypoint within a small margin."""
        for wp in waypoints:
            if math.hypot(wp[0] - point[0], wp[1] - point[1]) < threshold:
                return True
        return False

    def is_too_close_to_predefined(self, point, waypoints, threshold=0.5):
        """Check if the point is too close to any predefined waypoint (used for filtering)."""
        for wp in waypoints:
            if math.hypot(wp[0] - point[0], wp[1] - point[1]) < threshold:
                return True
        return False

    def set_waypoints(self, x, y):
        """Return a list of predefined waypoints offset by the car’s current position."""
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
        """
        Perform RRT* path planning from current position to the goal using waypoints and detected obstacles.

        Returns
        -------
        tuple of (x_vals, y_vals): Lists of x and y coordinates along the path.
        """
        if not self.path_found:
            x = self.localizer.x / 1000
            y = self.localizer.y / 1000

            start = [x, y]
            map_size = [20, 20]
            obstacles = self.set_obstacles(detections, x, y)

            path = []
            points = [start] + waypoints + [goal[:2]]

            for i in range(len(points) - 1):
                rrt_star = RRTStar(start=points[i], goal=points[i + 1],
                                   num_obstacles=0, map_size=map_size, obstacles=obstacles)
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

            # Filter nodes to remove ones too close to waypoints
            filtered_path = []
            for node in path:
                if self.is_actual_waypoint([node.x, node.y], waypoints) or not self.is_too_close_to_predefined(
                        [node.x, node.y], waypoints):
                    filtered_path.append(node)

            x_vals = [node.x if hasattr(node, "x") else node[0] for node in filtered_path]
            y_vals = [node.y if hasattr(node, "y") else node[1] for node in filtered_path]
            return x_vals, y_vals

    def set_goal(self, det, x, y, theta):
        """Set a goal based on a detected car ahead."""
        if det["class"] == "Car" and self.goal_set is False:
            distance_offset = 12.0  # Extra distance for stopping before the object
            x_goal = x
            y_goal = det["distance"] + distance_offset + y
            goal = (x_goal, y_goal, theta)
            print(f"Goal set to: {goal} at distance {det['distance']:.2f}m")
            return goal
        elif det["class"] == "Car" and self.goal_set is True:
            print("Goal is already set.")
        else:
            print("No suitable car detected to set goal.")
            return None

    def set_obstacles(self, detections, x, y):
        """Convert detected car objects into obstacle rectangles for path planning."""
        obstacles = []
        for det in detections:
            if det["class"] == "Car":
                obstacles.append((-1.25 + x, det["distance"] + y, 2.5, 4))  # width and height of car
        return obstacles

    def waypoint_reached(self, x_wp, y_wp, threshold=0.1):
        """Determine if a given waypoint has been reached."""
        current_x, current_y = self.localizer.x, self.localizer.y
        return math.hypot(x_wp - current_x, y_wp - current_y) < threshold

    def goal_reached(self, threshold=0.1):
        """Check if the final goal has been reached."""
        if not self.goal:
            return False
        goal_x, goal_y, _ = self.goal
        return self.waypoint_reached(goal_x, goal_y, threshold)

    def calculate_angle(self, x, y, waypoint):
        """Compute angle between current location and a waypoint."""
        dx = waypoint[0] - x
        dy = waypoint[1] - y
        return round(math.atan2(dy, dx))

    def angle_difference(self, desired_theta, current_theta):
        """Compute smallest angle difference (normalized)."""
        return -((desired_theta - current_theta + math.pi) % (2 * math.pi) - math.pi)

    def adjust_steering(self, steering_angle):
        """Scale and clamp steering value according to vehicle type."""
        if self.car_type == 'Hunter':
            new_steering_angle = round(steering_angle * 576 / 90)
            return max(min(new_steering_angle, 576), -576)
        else:
            new_steering_angle = steering_angle * (1.25 / 45)
            return max(min(new_steering_angle, 1.25), -1.25)

    def set_steering_angle(self, angle_difference, steering_angle=None):
        """Send steering command to the controller."""
        angle_in_degrees = math.degrees(angle_difference)
        new_steering_angle = self.adjust_steering(angle_in_degrees)
        print(f"Steering angle diff (deg): {angle_in_degrees:.2f}, Command: {new_steering_angle:.2f}")
        if self.car_type == 'Hunter':
            self.can_controller.set_steering_and_throttle(new_steering_angle, 300)
        else:
            self.can_controller.set_steering(steering_angle)

    def steer_toward_waypoint(self, waypoint):
        """Align vehicle’s heading toward the next waypoint."""
        x = self.localizer.x / 1000
        y = self.localizer.y / 1000
        theta = self.localizer.theta
        desired_theta = self.calculate_angle(x, y, waypoint)
        angle_diff = self.angle_difference(desired_theta, theta)
        self.set_steering_angle(angle_diff)

    def plot_waypoints(self, goal, detections, x_vals, y_vals):
        """Plot the generated RRT path and obstacles using matplotlib."""
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
        """Return the distance to the closest object using LIDAR scan data."""
        scan_data = np.full(360, np.inf)
        for _, angle, distance in scan:
            angle_idx = min(359, floor(angle))
            if distance < 150:
                scan_data[angle_idx] = np.inf
            else:
                scan_data[angle_idx] = distance

        closest_distance = np.min(scan_data)
        print(f"Closest object distance: {closest_distance:.2f} mm")
        return closest_distance

    def check_collision(self, closest_distance):
        """Stop the vehicle if an obstacle is dangerously close."""
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
        """
        Main logic loop: detects objects, sets goal, plans path, navigates, and avoids collision.
        Should be called repeatedly in real-time control loop.
        """
        _, detections, _ = self.object_detection.process(front_view)

        x = self.localizer.x / 1000
        y = self.localizer.y / 1000
        theta = self.localizer.theta

        for det in detections:
            if det["class"] == "Car" and det["distance"] <= 10 and not self.goal_set:
                self.goal = self.set_goal(det, x, y, theta)
                self.goal_set = True

                if self.goal:
                    waypoints = self.set_waypoints(x, y)
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

# Below is a manual implementation of the overtake not reliant on a localization
    def steer_to_centre(self, detections=None, center_x=None):
        """Steers the vehicle to the center of the lane based on lane and object detection.

        Parameters
        ----------
        detections : list[dict], optional
            List of detected objects including their class and bounding boxes.
        center_x : float, optional
            X-coordinate of the lane center in the camera frame.

        Returns
        -------
        bool
            True if the vehicle is centered, False otherwise.
        """

        if not detections is not None:
            print("No detections available for steering.")
            return False

        lane_width = 3  # meters
        scaling_factor = lane_width / 2
        lane_offset = (center_x - (CameraResolution.WIDTH / 2)) / CameraResolution.WIDTH * scaling_factor

        vehicle_offset = 0
        for det in detections:
            if det['class'] == "Car":
                x1, y1, x2, y2 = det['bbox']
                object_center_x = (x1 + x2) / 2
                screen_center_x = CameraResolution.WIDTH / 2
                vehicle_offset = object_center_x - screen_center_x

        lane_weight = 0.7
        vehicle_weight = 0.3
        combined_offset = (lane_weight * lane_offset + vehicle_weight * vehicle_offset)

        Kp = 0.5
        steering_angle = round(Kp * combined_offset)

        if self.car_type == 'Hunter':
            MAX_STEERING_ANGLE = 576
            steering_angle = max(round(min(steering_angle, MAX_STEERING_ANGLE), -MAX_STEERING_ANGLE))
        else:
            MAX_STEERING_ANGLE = 1.25
            steering_angle = max(min(steering_angle, MAX_STEERING_ANGLE), -MAX_STEERING_ANGLE)

        DEADZONE = 0.1 if self.car_type != 'Hunter' else 10
        if abs(steering_angle) < DEADZONE:
            print("Vehicle centered in lane. Stopping steering adjustments.")
            steering_angle = 0
            self.centered = True
        else:
            self.centered = False
            print(f"Steering {'right' if combined_offset > 0 else 'left'} by {steering_angle} CAN units.")
            print(f"Lane offset: {lane_offset:.2f}m, Vehicle offset: {vehicle_offset:.2f}px")

        if self.car_type == 'Hunter':
            self.can_controller.set_steering_and_throttle(steering_angle, 300)
        else:
            self.can_controller.set_steering(steering_angle)
            self.can_controller.set_throttle(50)

        self.centered = True
        return self.centered

    def iter_scans(self):
        """Generator that yields LIDAR scan data in chunks.

        Yields
        ------
        list[tuple[float, float]]
            A list of tuples containing angle and distance from LIDAR.
        """
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
        """Fetches a single short LIDAR scan.

        Returns
        -------
        list[tuple[float, float]]
            A list of (angle, distance) tuples.
        """
        scan_data = []
        for _, angle, distance in self.lidar.iter_measures():
            if len(scan_data) > 5:
                return scan_data
            scan_data.append((angle, distance))

    def check_collision_during_overtake(self):
        """Checks for obstacles during an overtaking maneuver using LIDAR.

        Returns
        -------
        bool
            True if a potential collision is detected, False otherwise.
        """
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
        """Performs a scan to the right to determine if the overtaken car has been passed."""
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
        """Main loop handling manual control and overtaking sequence.

        Parameters
        ----------
        front_view : Any, optional
            The camera input frame used for object detection and lane centering.
        """
        _, detections, _ = self.object_detection.process(front_view)
        print(f"Detections type: {type(detections)}")
        print(f"Detections content: {detections}")

        current_time = time.time()

        if not self.steering_state:
            # Attempt to center the car
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
            # Begin overtaking maneuver
            if self.car_type == 'Hunter':
                self.can_controller.set_steering_and_throttle(-100, 300)
            else:
                self.can_controller.set_kart_gearbox(KartGearBox.forward)
                self.can_controller.set_throttle(50)
                self.can_controller.set_steering(-1.25)

            if self.check_collision_during_overtake():
                self.steering_state = None
                return
            self.steering_state = 'Left'
            self.start_timer = current_time

            if self.steering_state == 'Left' and (current_time - self.start_timer) >= 1:
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