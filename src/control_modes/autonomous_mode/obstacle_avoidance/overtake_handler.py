from ..object_detection.Detection import ObjectDetection
from ..old_twente_code.can_listener import CanListener
from .....src.misc import calculate_steering, calculate_throttle

class OvertakeHandler(ObjectDetection):
    def __init__(self, weights_path, input_source):
        super().__init__(weights_path, input_source)
        self.car_distance_threshold = 2
        self.car_detected = False
        self.overtake_complete = False

    def detect_car(self, detections):
        for det in detections:
            if det["class"] == "Car":
                self.car_detected = True
                return True
        return False

    def initialize_overtake(self, detections):
        if self.car_detected:
            for det in detections:
                if det["class"] == "Car":
                    x1, y1, x2, y2 = det['bbox']
                    distance = self.estimate_distance(x1, y1, x2, y2, "Car")
                    if distance < self.car_distance_threshold:
                        print("Car detected")
                        return True
            return False