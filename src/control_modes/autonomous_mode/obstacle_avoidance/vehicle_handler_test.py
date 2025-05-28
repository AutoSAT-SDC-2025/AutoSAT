from .RRT import RRTStar
from ..object_detection.Detection import ObjectDetection
import cv2
import math
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt

class VehicleHandler:
    def __init__(self, weights_path, input_source):
        self.detect_objects = ObjectDetection(weights_path, input_source)
        self.cam = cv2.VideoCapture("assets/CarRecording480.mp4")
        self.goal_set = False
        self.path_found = False

    def set_rrt(self, goal, detections, waypoints):
        if not self.path_found:
            start = [0, 0]
            map_size = [20, 20]
            obstacles = self.set_obstacles(detections)

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

    def set_goal(self, det):
        if det["class"] == "Car" and not self.goal_set:
            distance_offset = 12.0
            goal = (0, det["distance"] + distance_offset, 0)
            print(f"Goal set to: {goal} at distance {det['distance']:.2f}m")
            return goal

    def set_obstacles(self, detections):
        obstacles = []
        for det in detections:
            if det["class"] == "Car":
                obstacles.append((-1.25, det["distance"], 2.5, 4))
        return obstacles

    def plot_waypoints(self, goal, detections, x_vals, y_vals):
        obstacles = self.set_obstacles(detections)
        plt.figure(figsize=(8, 8))
        plt.plot(x_vals, y_vals, marker='o', color='blue', label='RRT* Path')
        plt.scatter([x_vals[0]], [y_vals[0]], color='green', label='Start')
        plt.scatter([goal[0]], [goal[1]], color='red', label='Goal')

        # Plot obstacles (optional, if you want to visualize them)
        for ox, oy, width, height in obstacles:
            rect = plt.Rectangle((ox, oy), width, height, color='gray', alpha=0.5)
            plt.gca().add_patch(rect)

        plt.xlim(-5, 5)
        plt.ylim(0, 25)
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

            detections = self.detect_objects.detect_objects(frame)

            for det in detections:
                x1, y1, x2, y2 = det['bbox']
                label = f"{det['class']} ({det['distance']:.2f}m)"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                if det["class"] == "Car" and det["distance"] <= 8 and not self.goal_set:
                    self.goal = self.set_goal(det)
                    self.goal_set = True
                    if self.goal:
                        waypoints = self.set_waypoints()
                        result = self.set_rrt(self.goal, detections, waypoints)
                        if result:
                            self.x_vals, self.y_vals = result
                            self.plot_waypoints(self.goal, detections, self.x_vals, self.y_vals)

            cv2.imshow('Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cam.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    weights_path = "assets/v5_model.pt"
    input_source = "video"
    vehicle_handler = VehicleHandler(weights_path, input_source)
    vehicle_handler.main()
