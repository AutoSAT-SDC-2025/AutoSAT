from rplidar import RPLidar
import cv2
from ....can_interface.bus_connection import connect_to_can_interface, disconnect_from_can_interface
from ....can_interface.can_factory import select_can_controller_creator, create_can_controller
from src.util.video import get_camera_config
from ..object_detection.Detection import ObjectDetection

class PedestrianHandler:
    def __init__(self, weights_path, input_source):
        self.captures = None
        self.car_type = "Hunter"
        self.can_bus = connect_to_can_interface(0)

        self.can_creator = select_can_controller_creator(self.car_type)
        self.can_controller = create_can_controller(self.can_creator, self.can_bus)
        self.cams = get_camera_config()
        try:
            self.lidar = RPLidar("Com3")
        except:
            self.lidar = RPLidar("/dev/ttyUSB0")
        self.object_detection = ObjectDetection(weights_path, input_source)
        self.cam = cv2.VideoCapture(1)
        self.person_distance_threshold = 2
        self.previous_detection = {}
        self.focal_length = 540
        self.image_height = 480
        self.image_width = 848
        self.initial_position = None
        self.current_position = None
        self.direction = None

    def detect_objects(self):
        ret, frame = self.cam.read()
        if not ret:
            print("Failed to capture frame from camera")
            return []
        return self.object_detection.detect_objects(frame)

    def get_initial_position(self, detections):
        for det in detections:
            if det["class"] == "Person":
                x1, _, x2, _ = det["bbox"]
                if (x1 + x2) / 2 < self.image_width / 2:
                    self.initial_position = "Left"
                else:
                    self.initial_position = "Right"

    def get_direction(self, detections):
        for det in detections:
            if det["class"] == "Person":
                x1, y1, x2, y2 = det["bbox"]
                x_center = (x1 + x2) / 2
                obj_id = "Person"

                if obj_id in self.previous_detection:
                    prev_x_center = self.previous_detection[obj_id]["x_center"]

                    if x_center > prev_x_center + 2:
                        print("Pedestrian is going right")
                        self.direction = "Right"
                    elif x_center < prev_x_center - 2:
                        print("Pedestrian is going left")
                        self.direction = "Left"
                    else:
                        print("Pedestrian is stationary")
                        self.direction = "Stationary"
                else:
                    self.direction = "Unknown"

                self.previous_detection[obj_id] = {"x_center": x_center}

        return self.direction

    def get_current_pos(self, detections):
        for det in detections:
            if det["class"] == "Person":
                x_center = (det["bbox"][0] + det["bbox"][2]) / 2
                print(x_center)
                if x_center < self.image_width / 2:
                    self.current_position = "Left"
                elif x_center > self.image_width / 2:
                    self.current_position = "Right"
                else:
                    self.current_position = "Unknown"
        return self.current_position

    def pedestrian_crossed(self):
        if self.current_position == "Left" and self.initial_position == "Right" and self.direction == "Stationary":
            return True
        elif self.current_position == "Right" and self.initial_position == "Left" and self.direction == "Stationary":
            return True
        return False

    def stop_car(self, object_detections):
        for obj in object_detections:
            if obj["class"] == "Person" and 0 < obj["distance"] < self.person_distance_threshold:
                self.can_controller.set_steering_and_throttle(0, 0)
                self.can_controller.set_parking_mode(1)
                print("Stopped for pedestrian")
                return True
        self.can_controller.set_parking_mode(0)
        self.can_controller.set_steering_and_throttle(0, 300)
        return "Continuing to drive"

    def continue_driving(self):
        if self.pedestrian_crossed():
            print("Continuing driving")
            ped_parking_mode = self.can_controller.set_parking_mode(0)
            ped_driving_mode = self.can_controller.set_steering_and_throttle(0, 300)
            return ped_driving_mode, ped_parking_mode

    def main(self):
        initial_position_set = False

        while True:
            detections = self.detect_objects()
            if not detections:
                continue

            if not initial_position_set:
                self.get_initial_position(detections)
                initial_position_set = True

            self.get_direction(detections)
            self.get_current_pos(detections)

            status = self.stop_car(detections)
            print(status)

            if self.pedestrian_crossed():
                drive_status = handler.continue_driving()
                print("Pedestrian crossed. Continuing to drive:", drive_status)

if __name__ == "__main__":
    weights_path = "assets/v5_model.pt"
    input_source = "video"
    handler = PedestrianHandler(weights_path, input_source)
    handler.main()