from ..object_detection.Detection import ObjectDetection
from .RRT import RRTStar

import cv2
import torch
import matplotlib

matplotlib.use('TkAgg')

import matplotlib.pyplot as plt

class RRTSim:
    def __init__(self, weights_path, input_source):
        self.object_detection = ObjectDetection(weights_path, input_source)
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path="assets/v5_model.pt", force_reload=True)
        self.cam = cv2.VideoCapture("assets/CarCropped.mp4")
        self.goal_set = False
        self.path_found = False
        self.goal = None
        self.detections = None
        self.x_vals = None
        self.y_vals = None

    def open_camera(self):
        while self.cam.isOpened():
            ret, frame = self.cam.read()
            if not ret:
                break

            traffic_state, detections = self.object_detection.process(frame)

            for det in detections:
                x1, y1, x2, y2 = det['bbox']
                label = f"{det['class']} ({det['distance']:.2f}m)"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                if det["class"] == "Car" and det["distance"] <= 5 and self.goal_set is False:
                    self.goal = self.set_goal(det)
                    self.goal_set = True
                    if self.goal:
                        result = self.set_rrt(self.goal, detections)
                        if result:
                            self.x_vals, self.y_vals = result
                            self.detections = detections

            cv2.imshow('Detection', frame)

            if cv2.waitKey(1) & 0xFF == ord('q') or (self.goal_set and self.path_found):
                break

        self.cam.release()
        cv2.destroyAllWindows()

        if self.goal and self.detections and self.x_vals and self.y_vals:
            self.plot_waypoints(self.goal, self.detections, self.x_vals, self.y_vals)

    def set_goal(self, det):
        if det["class"] == "Car" and self.goal_set is False:
            distance_offset = 8.0
            x_goal = 0
            y_goal = det["distance"] + distance_offset
            goal = (x_goal, y_goal)
            print(f"Goal set to: {goal} at distance {det['distance']:.2f}m")
            return goal
        elif det["class"] == "Car" and self.goal_set is True:
            print("Goal is already set.")
        else:
            print("No suitable car detected to set goal.")
            return None

    def set_obstacles(self, detections):
        obstacles = []
        for det in detections:
            if det["class"] == "Car":
                ox = -1.25
                oy = det["distance"]
                width = 2.5
                height = 4
                obstacles.append((ox, oy, width, height))
        return obstacles

    def set_rrt(self, goal, detections):
        if self.path_found is False:
            start = [0, 0]
            map_size = [20, 20]

            obstacles = self.set_obstacles(detections)

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
        plt.ylim(0, 15)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('RRT* Path Planning')
        plt.legend()
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    weights_path = "assets/v5_model.pt"
    input_source = "video"
    handler = RRTSim(weights_path, input_source)
    handler.open_camera()

