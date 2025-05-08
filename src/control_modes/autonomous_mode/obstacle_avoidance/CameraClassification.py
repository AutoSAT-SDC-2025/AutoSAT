import cv2

from .lidar_scan import LidarScans
from ..object_detection.Detection import ObjectDetection
from .rrttestright import RRTStar, Node

from rplidar import RPLidar
import math
import torch
import numpy as np

class ScanMerging:
    def __init__(self, weights_path, input_source):
        self.object_detection = ObjectDetection(weights_path, input_source)
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path="assets/v5_model.pt", force_reload=True)
        self.cam = cv2.VideoCapture(1)
        self.lidar = RPLidar("COM3")
        self.lidar_scans = LidarScans()
        self.image_width = 1920
        self.image_height = 1080
        self.focal_length = 540

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
        x = (distance / 1000.0) * math.cos(radians)
        y = (distance / 1000.0) * math.sin(radians)
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

    def overlay_on_camera(self):
        for scan in self.iter_scans():
            ret, frame = self.cam.read()
            if not ret:
                break

            detections = self.object_detection.detect_objects(frame)
            merged = self.merge_measurements(detections, scan)

            for angle, distance in scan:
                if distance < 100:
                    continue

                x, y = self.coordinate_conversion(angle, distance)
                u, v = self.homogeneous_coordinates(x, y)

                if 0 <= int(u) < self.image_width and 0 <= int(v) < self.image_height:
                    cv2.circle(frame, (int(u), int(v)), 3, (0, 255, 0), -1)

            for obj in merged:
                x1, y1, x2, y2 = map(int, obj["box"])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                label = f"{obj['class']} {obj['distance']:.2f}m"
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            cv2.imshow("LIDAR + Detection Overlay", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    weights_path = "assets/yolov5s.pt"
    input_source = "video"
    handler = ScanMerging(weights_path, input_source)
    handler.overlay_on_camera()
    for scan in handler.iter_scans():
        ret, frame = handler.cam.read()
        if not ret:
            print("Camera frame could not be retrieved. Exiting...")
            break
        detections = handler.object_detection.detect_objects(frame)
        merged_measurements = handler.merge_measurements(detections, scan)
        for obj in merged_measurements:
            print(f"Class: {obj['class']}, "
                  f"Distance: {obj['distance']:.2f}, "
                  f"Box: {obj['box']}")

    handler.cam.release()
    handler.lidar.stop()
    handler.lidar.disconnect()
