from ..object_detection.Detection import ObjectDetection

import threading
import cv2
from rplidar import RPLidar
import math
import torch
import numpy as np
import matplotlib

matplotlib.use('TKAgg')

class SensorFusion:
    """
    A class that fuses object detection from a camera with distance data from a LiDAR sensor.

    Attributes:
        object_detection (ObjectDetection): Instance for detecting objects from video input.
        cam (cv2.VideoCapture): OpenCV camera capture object.
        model (torch.nn.Module): YOLOv5 model loaded with custom weights.
        lidar (RPLidar): RPLidar object for LiDAR data acquisition.
        image_width (int): Width of the camera image.
        image_height (int): Height of the camera image.
        focal_length (int): Focal length used for camera intrinsics.
        latest_scan (list): Most recent LiDAR scan data.
        scan_lock (threading.Lock): Lock for thread-safe access to scan data.
        lidar_thread (threading.Thread): Background thread for LiDAR data gathering.
    """

    def __init__(self, weights_path, input_source):
        """
        Initializes the SensorFusion system by loading the detection model, camera, and LiDAR.

        Args:
            weights_path (str): Path to the YOLOv5 model weights.
            input_source (str): Input source descriptor (e.g., 'video').
        """
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
        """
        Continuously runs in a background thread to collect LiDAR measurements.

        Filters out zero or very short-range readings (< 300mm) and stores a 360-point scan.
        """
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
        """
        Retrieves the most recent full 360-degree LiDAR scan.

        Returns:
            list: A list of (angle, distance) tuples.
        """
        with self.scan_lock:
            return list(self.latest_scan)

    def coordinate_conversion(self, angle, distance, angle_offset=90):
        """
        Converts polar LiDAR coordinates to Cartesian coordinates in meters.

        Args:
            angle (float): Angle in degrees from LiDAR.
            distance (float): Distance in millimeters from LiDAR.
            angle_offset (float, optional): Offset to align LiDAR with camera view. Defaults to 90.

        Returns:
            tuple: (x, y) Cartesian coordinates in meters.
        """
        corrected_angle = angle + angle_offset
        radians = math.radians(corrected_angle)
        x = (distance / 1000.0) * math.cos(radians)
        y = (distance / 1000.0) * math.sin(radians)
        return x, y

    def homogeneous_coordinates(self, x_lidar, y_lidar):
        """
        Transforms 2D LiDAR coordinates to 3D camera space and projects to 2D image space.

        Args:
            x_lidar (float): X coordinate in meters from LiDAR.
            y_lidar (float): Y coordinate in meters from LiDAR.

        Returns:
            tuple: (u, v) pixel coordinates on the image plane.
        """
        point_lidar = np.array([[x_lidar], [y_lidar], [0], [1]])

        R_base = np.array([
            [1, 0, 0],
            [0, 0, 1],
            [0, -1, 0]
        ])

        angle = np.deg2rad(-20)
        R_tilt = np.array([
            [1, 0, 0],
            [0, np.cos(angle), -np.sin(angle)],
            [0, np.sin(angle), np.cos(angle)]
        ])

        R = R_tilt @ R_base
        t = np.array([[-0.3], [0], [0.5]])

        T = np.vstack((np.hstack((R, t)), np.array([[0, 0, 0, 1]])))
        point_cam_homogeneous = T @ point_lidar
        point_cam = point_cam_homogeneous[:3]

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
        """
        Displays a live video feed with:
            - Bounding boxes from object detection.
            - Projected LIDAR points.
            - Estimated distances to detected objects.
        Press 'q' to exit.
        """
        while self.cam.isOpened():
            ret, frame = self.cam.read()
            if not ret:
                break

            scan = self.get_latest_scan()
            results = self.model(frame)
            detections = results.xyxy[0].cpu().numpy()
            bounding_boxes = []
            u_temp, v_temp = [], []

            for angle, distance in scan:
                if distance < 300:
                    continue
                x, y = self.coordinate_conversion(angle, distance)
                u, v = self.homogeneous_coordinates(x, y)
                u_temp.append(u)
                v_temp.append(v)

            for det in detections:
                x1, y1, x2, y2, conf, class_id = det[:6]
                class_name = self.model.names[int(class_id)]

                bounding_boxes.append((x1, y1, x2, y2))
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                label = f"{class_name} {conf:.2f}"
                cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            for (angle, distance), u, v in zip(scan, u_temp, v_temp):
                if 0 <= u < frame.shape[1] and 0 <= v < frame.shape[0]:
                    inside_box = any(x1 <= u <= x2 and y1 <= v <= y2 for (x1, y1, x2, y2) in bounding_boxes)
                    color = (0, 255, 0) if inside_box else (0, 0, 255)
                    cv2.circle(frame, (u, v), 5, color, -1)

            for det in detections:
                x1, y1, x2, y2, conf, class_id = det[:6]
                class_name = self.model.names[int(class_id)]

                distances_in_box = [distance / 1000.0 for (angle, distance), u, v in zip(scan, u_temp, v_temp)
                                    if x1 <= u <= x2 and y1 <= v <= y2]

                if distances_in_box:
                    estimated_distance = np.median(distances_in_box)
                    if estimated_distance < 0.3:
                        continue
                    print(f"Estimated distance to {class_name}: {estimated_distance:.2f} m")
                    cv2.putText(frame, f"{estimated_distance:.2f} m",
                                (int(x1), int(y2) + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            cv2.imshow("YOLO Detection Overlay", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cam.release()
        cv2.destroyAllWindows()
        self.lidar.stop()
        print("Stopping Lidar")
        self.lidar.disconnect()

    def get_frame_with_detections(self):
        """
        Captures a frame and returns it with structured object detection information.

        Returns:
            tuple:
                - frame (np.ndarray): Captured video frame.
                - annotated_objects (list): List of dicts with detection data:
                    {
                        "bbox": (x1, y1, x2, y2),
                        "class": str,
                        "confidence": float,
                        "distance": float (in meters, or -1 if undetected)
                    }
        """
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
            x, y = self.coordinate_conversion(angle, distance)
            u, v = self.homogeneous_coordinates(x, y)
            u_temp.append(u)
            v_temp.append(v)

        annotated_objects = []
        for det in detections:
            x1, y1, x2, y2, conf, class_id = det[:6]
            class_name = self.model.names[int(class_id)]

            distances_in_box = [distance / 1000.0 for (angle, distance), u, v in zip(scan, u_temp, v_temp)
                                if x1 <= u <= x2 and y1 <= v <= y2]

            estimated_distance = np.median(distances_in_box) if distances_in_box else -1

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
    handler = SensorFusion(weights_path, input_source)
    handler.overlay_on_camera()
