import threading

import cv2
from IPython.core.pylabtools import figsize

from ..object_detection.Detection import ObjectDetection

from rplidar import RPLidar
import math
import torch
import numpy as np
import matplotlib

matplotlib.use('TKAgg')
import matplotlib.pyplot as plt


class ScanMerging:
    def __init__(self, weights_path, input_source):
        self.object_detection = ObjectDetection(weights_path, input_source)
        self.cam = cv2.VideoCapture(0)
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path="assets/yolov5s.pt", force_reload=True)
        self.lidar = RPLidar("COM3")
        self.image_width = 848
        self.image_height = 480
        self.focal_length = 540

        self.latest_scan = []
        self.scan_lock = threading.Lock()
        self.lidar_thread = threading.Thread(target=self.scan_loop)
        self.lidar_thread.daemon = True
        self.lidar_thread.start()

    """def iter_scans(self):
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
            self.lidar.disconnect()"""

    def scan_loop(self):
        scan = []
        try:
            for _, quality, angle, distance in self.lidar.iter_measures():
                if distance == 0 or distance > 4000:
                    continue
                scan.append((angle, distance))
                if len(scan) >= 360:
                    with self.scan_lock:
                        self.latest_scan = scan
                    scan = []
        except Exception as e:
            print("LIDAR scan error:", e)

    def get_latest_scan(self):
        with self.scan_lock:
            return list(self.latest_scan)

    def coordinate_conversion(self, angle, distance, angle_offset=90):
        corrected_angle = angle + angle_offset  # or -angle_offset, depending on rotation direction
        radians = math.radians(corrected_angle)
        x = (distance / 1000.0) * math.cos(radians)
        y = (distance / 1000.0) * math.sin(radians)
        return x, y

    def homogeneous_coordinates(self, x_lidar, y_lidar):
        x_rotated = x_lidar
        y_rotated = -y_lidar

        X_cam = x_rotated
        Y_cam = 0.1
        Z_cam = y_rotated - 0

        point_cam = np.array([[X_cam], [Y_cam], [Z_cam]])

        k = np.array([
            [self.focal_length, 0, self.image_width / 2],
            [0, self.focal_length, self.image_height / 2],
            [0, 0, 1]
        ])

        projected = k @ point_cam

        u = projected[0][0] / projected[2][0]
        v = projected[1][0] / projected[2][0]

        return int(u), int(v)

    def overlay_on_camera(self):
        while self.cam.isOpened():
            ret, frame = self.cam.read()
            if not ret:
                break
            # u_temp, v_temp = [], []
            scan = self.get_latest_scan()
            # First, detect and store bounding boxes
            results = self.model(frame)
            detections = results.xyxy[0].cpu().numpy()
            bounding_boxes = []
            for det in detections:
                x1, y1, x2, y2, conf, class_id = det[:6]

            u_temp, v_temp = [], []

            # Then, draw lidar points with color depending on whether they are inside any bbox
            for angle, distance in scan:
                x, y = self.coordinate_conversion(angle, distance, angle_offset=90)
                u, v = self.homogeneous_coordinates(x, y)

                u_temp.append(u)
                v_temp.append(v)

                if 0 <= u < frame.shape[1] and 0 <= v < frame.shape[0]:
                    inside_box = False
                    for (x1, y1, x2, y2) in bounding_boxes:
                        if x1 <= u <= x2 and y1 <= v <= y2:
                            inside_box = True
                            break
                    color = (0, 255, 0) if inside_box else (0, 0, 255)  # Green if inside, Red if outside
                    cv2.circle(frame, (u, v), 5, color, -1)

            # results = self.model(frame)
            # detections = results.xyxy[0].cpu().numpy()

            for det in detections:
                x1, y1, x2, y2, conf, class_id = det[:6]
                class_name = self.model.names[int(class_id)]

                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                label = f"{class_name} {conf:.2f}"
                cv2.putText(frame, label, (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                distances_in_box = []
                for (angle, distance), u, v in zip(scan, u_temp, v_temp):
                    if x1 <= u <= x2:
                        print(f"u: {u}, x1: {x1}, x2: {x2}")  # Print u with bounding box edges
                        distances_in_box.append(distance / 1000.0)

                if distances_in_box:
                    estimated_distance = np.median(distances_in_box)
                    print(f"Estimated distance to {class_name}: {estimated_distance:.2f} m")

                    cv2.putText(frame, f"{estimated_distance:.2f} m",
                                (int(x1), int(y2) + 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            cv2.imshow("YOLO Detection Overlay", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cam.release()
        cv2.destroyAllWindows()
        self.lidar.stop()
        print("Stopping Lidar")
        self.lidar.disconnect()


if __name__ == '__main__':
    weights_path = "assets/yolov5s.pt"
    input_source = "video"
    handler = ScanMerging(weights_path, input_source)
    handler.overlay_on_camera()