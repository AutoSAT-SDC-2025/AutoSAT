import cv2
from ..object_detection.Detection import ObjectDetection

class PedestrianHandler:
    def __init__(self, weights_path, input_source):
        self.detector = ObjectDetection(weights_path, input_source)
        self.cam = cv2.VideoCapture("assets/PersonRecording480.mp4")
        self.person_distance_threshold = 7
        self.previous_detection = {}
        self.focal_length = 540
        self.image_height = 480
        self.image_width = 852
        self.initial_position = None
        self.current_position = None
        self.direction = None
        self.cross_counter = 0
        self.cross_threshold = 5
        self.crossed_flag = False

    def detect_objects(self, frame):
        return self.detector.detect_objects(frame)

    def get_initial_position(self, detections):
        for det in detections:
            if det["class"] == "Person":
                x1, _, x2, _ = det["bbox"]
                if (x1 + x2) / 2 < self.image_width/2:
                    self.initial_position = "Left"
                elif (x1 + x2) / 2 > self.image_width/2:
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
                if x_center < self.image_width/2:
                    self.current_position = "Left"
                if x_center > self.image_width/2:
                    self.current_position = "Right"
        return self.current_position

    def pedestrian_crossed(self):
        if self.crossed_flag:
            return False

        if ((self.current_position == "Left" and self.initial_position == "Right" and self.direction == "Stationary") or
            (self.current_position == "Right" and self.initial_position == "Left" and self.direction == "Stationary")):

            self.cross_counter += 1
            print(f"Crossing counter: {self.cross_counter}")

            if self.cross_counter >= self.cross_threshold:
                print("Pedestrian has crossed")
                self.crossed_flag = True
                return True
        else:
            self.cross_counter = 0

        return False

    def main(self):
        initial_position_set = False

        while self.cam.isOpened():
            ret, frame = self.cam.read()
            if not ret:
                break

            detections = self.detect_objects(frame)

            for det in detections:
                if det["class"] == "Person":
                    x1, y1, x2, y2 = [int(coord) for coord in det["bbox"]]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f'{det["class"]}'
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            if detections:
                for det in detections:
                    if det["class"] == "Person":
                        x1, y1, x2, y2 = [int(coord) for coord in det["bbox"]]
                        distance = det.get("distance", None)

                        # Draw bounding box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                        # Draw label and distance
                        label = f'{det["class"]} - {distance:.2f}m' if distance is not None else det["class"]
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                        # Print stop condition if person is close
                        if distance is not None and distance <= self.person_distance_threshold:
                            print("Stopping car - pedestrian at {:.2f}m".format(distance))

                if not initial_position_set:
                    self.get_initial_position(detections)
                    initial_position_set = True
                    print(f"Initial position set to: {self.initial_position}")

                self.get_direction(detections)
                self.get_current_pos(detections)

                print(
                    f"Initial: {self.initial_position}, Current: {self.current_position}, Direction: {self.direction}")

                self.pedestrian_crossed()

                if self.crossed_flag:
                    cv2.putText(frame, "Pedestrian has crossed", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

            # Show the frame
            cv2.imshow("Pedestrian Detection", frame)

            # Exit on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cam.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    weights_path = "assets/v5_model.pt"
    input_source = "video"
    handler = PedestrianHandler(weights_path, input_source)
    handler.main()