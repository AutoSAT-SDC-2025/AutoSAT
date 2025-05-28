from ..object_detection.Detection import ObjectDetection

import threading
import cv2
from rplidar import RPLidar
import math
import torch
import numpy as np
import matplotlib

matplotlib.use('TKAgg')

class ScanMerging:
    def __init__(self, weights_path, input_source):
        self.object_detection = ObjectDetection(weights_path, input_source)
        self.cam = cv2.VideoCapture(1)
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path="assets/v5_model.pt", force_reload=True)
        self.lidar = RPLidar("COM3")
        self.image_width = 848
        self.image_height = 480
        self.focal_length = 540

        self.latest_scan = []
        self.scan_lock = threading.Lock()
        self.lidar_thread = threading.Thread(target=self.scan_loop)
        self.lidar_thread.daemon = True
        self.lidar_thread.start()

    def scan_loop(self):
        scan = []
        try:
            for _, quality, angle, distance in self.lidar.iter_measures():
                if distance == 0 or distance < 300:
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
        # 3D point in LiDAR coordinate frame (assume z = 0 for 2D lidar)
        point_lidar = np.array([[x_lidar], [y_lidar], [0], [1]])

        # Base rotation matrix for LiDAR to camera alignment
        R_base = np.array([
            [1, 0, 0],
            [0, 0, 1],  # Swapped 2nd and 3rd axes
            [0, -1, 0]
        ])

        # Additional rotation matrix for -20 degree tilt around X-axis
        angle = np.deg2rad(-20)  # Convert -20 degrees to radians
        R_tilt = np.array([
            [1, 0, 0],
            [0, np.cos(angle), -np.sin(angle)],
            [0, np.sin(angle), np.cos(angle)]
        ])

        # Combine rotations
        R = R_tilt @ R_base

        # Translation vector (camera is 0.3m in front, 0.5m below LiDAR)
        t = np.array([[-0.3], [0], [0.5]])

        # Homogeneous transformation matrix (4x4)
        T = np.vstack((np.hstack((R, t)), np.array([[0, 0, 0, 1]])))

        # Transform LiDAR point to camera coordinates
        point_cam_homogeneous = T @ point_lidar
        point_cam = point_cam_homogeneous[:3]  # extract x, y, z

        # Camera intrinsic matrix
        k = np.array([
            [self.focal_length, 0, self.image_width / 2],
            [0, self.focal_length, self.image_height / 2],
            [0, 0, 1]
        ])

        # Project to image plane
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
                if distance < 300:
                    continue
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
                    if estimated_distance < 0.3:
                        continue
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

    def get_frame_with_detections(self):
        ret, frame = self.cam.read()
        if not ret:
            return None, []

        scan = self.get_latest_scan()
        results = self.model(frame)
        detections = results.xyxy[0].cpu().numpy()

        u_temp, v_temp = [], []
        for angle, distance in scan:
            if distance < 300:
                continue
            x, y = self.coordinate_conversion(angle, distance, angle_offset=90)
            u, v = self.homogeneous_coordinates(x, y)
            u_temp.append(u)
            v_temp.append(v)

        annotated_objects = []
        for det in detections:
            x1, y1, x2, y2, conf, class_id = det[:6]
            class_name = self.model.names[int(class_id)]

            distances_in_box = []
            for (angle, distance), u, v in zip(scan, u_temp, v_temp):
                if x1 <= u <= x2 and y1 <= v <= y2:
                    distances_in_box.append(distance / 1000.0)

            if distances_in_box:
                estimated_distance = np.median(distances_in_box)
            else:
                estimated_distance = -1

            annotated_objects.append({
                "bbox": (x1, y1, x2, y2),
                "class": class_name,
                "confidence": conf,
                "distance": estimated_distance
            })

        return frame, annotated_objects

if __name__ == '__main__':
    weights_path = "assets/v5_model.pt"
    input_source = "video"
    handler = ScanMerging(weights_path, input_source)
    handler.overlay_on_camera()