import cv2
import numpy as np
import torch
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.general import non_max_suppression, scale_boxes, check_img_size
from yolov5.utils.torch_utils import select_device

from .TrafficDetection import TrafficManager

class ObjectDetection:
    """
    Handles object detection using a YOLOv5 model.
    Provides methods to detect objects, estimate their distance, and process results for downstream use.
    """
    def __init__(self, weights_path: str, input_source: str = 'video'):
        # Select device (CPU or GPU) for inference
        self.device = select_device('cpu')
        # Load YOLOv5 model with specified weights
        self.model = DetectMultiBackend(weights_path, device=self.device, fp16=False)
        self.model.warmup()
        self.input_source = input_source  # 'video' or 'camera'

        # Real-world dimensions (width, height in meters) for each class, used for distance estimation
        self.real_dims = {
            'Traffic Light Red': (0.07, 0.3),
            'Traffic Light Green': (0.07, 0.3),
            'Speed-limit-10km-h': (0.6, 0.6),
            'Speed-limit-15km-h': (0.6, 0.6),
            'Speed-limit-20km-h': (0.6, 0.6),
            'Car': (1.7, 1.5),
            'Person': (0.5, 1.9),
            'Object': (0.5, 0.5),
        }

        # Mapping from class indices to human-readable labels
        self.labels = {
            0: 'Car',
            1: 'Person',
            2: 'Speed-limit-10km-h',
            3: 'Speed-limit-15km-h',
            4: 'Speed-limit-20km-h',
            5: 'Traffic Light Green',
            6: 'Traffic Light Red'
        }
        # TrafficManager handles logic for interpreting traffic-related detections
        self.traffic_manager = TrafficManager()

    def estimate_distance(self, x1, y1, x2, y2, class_label):
        """
        Estimate the distance to an object using its bounding box and known real-world size.
        Uses a simple pinhole camera model.
        """
        focal_length = 540  # Camera focal length in pixels (empirically determined)
        real_width, real_height = self.real_dims.get(class_label, (0.5, 0.5))
        box_width, box_height = abs(x2 - x1), abs(y2 - y1)
        if box_width <= 0 or box_height <= 0:
            return 0
        # Average the real and pixel dimensions for a rough estimate
        distance = (focal_length * (real_width + real_height) / 2) / ((box_width + box_height) / 2)
        # Subtract a small offset to compensate for camera placement
        return max(distance - 0.3, 0)

    def detect_objects(self, frame):
        """
        Run object detection on a frame and return a list of detected objects with class, confidence, bbox, and distance.
        """
        # Ensure input image size is compatible with the model
        img_size = check_img_size(frame.shape[:2], s=32)
        frame_resized = cv2.resize(frame, (img_size[1], img_size[0]))
        img = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

        # Prepare image for model input (NCHW, float32, normalized)
        img = img.transpose((2, 0, 1))
        img = torch.from_numpy(np.expand_dims(img, axis=0)).float() / 255.0
        img = img.to(self.device)

        # Run inference
        with torch.no_grad():
            pred = self.model(img)

        # Apply non-max suppression to filter overlapping detections
        det = non_max_suppression(pred, 0.7, 0.45, max_det=1000)[0]
        detections = []
        if det is not None and len(det):
            # Scale bounding boxes back to original image size
            det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], frame.shape).round()
            for *xyxy, conf, cls in reversed(det):
                x1, y1, x2, y2 = map(int, xyxy)
                class_id = int(cls)
                class_label = self.labels.get(class_id, 'Unknown')
                if class_label in self.real_dims:
                    distance = self.estimate_distance(x1, y1, x2, y2, class_label)
                    detections.append({
                        'class': class_label,
                        'confidence': conf.item(),
                        'bbox': [x1, y1, x2, y2],
                        'distance': distance
                    })
        return detections

    def process(self, frame):
        """
        Detect objects in the frame, generate drawing instructions, and process traffic state.
        Returns:
            - traffic_state: dict with traffic light and speed limit info
            - detections: list of detected objects with metadata
            - draw_instructions: list of drawing commands for visualization
        """
        draw_instructions = []
        detections = self.detect_objects(frame)
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            class_label = det['class']

            # Add rectangle for each detection
            draw_instructions.append({
                'type': 'rect',
                'top_left': (x1, y1),
                'bottom_right': (x2, y2),
                'color': (0, 255, 0),
                'thickness': 2
            })

            # Add label and distance text
            draw_instructions.append({
                'type': 'text',
                'text': f"{class_label} ({det['distance']:.2f}m)",
                'position': (x1, y1 - 10),
                'font': 'FONT_HERSHEY_SIMPLEX',
                'font_scale': 0.6,
                'color': (0, 255, 0),
                'thickness': 2
            })
        # Analyze detections for traffic light and speed limit state
        traffic_state = self.traffic_manager.process_traffic_signals(detections)
        return traffic_state, detections, draw_instructions