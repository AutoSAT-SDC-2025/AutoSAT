from ..object_detection.Detection import ObjectDetection

class PedestrianHandler(ObjectDetection):
    def __init__(self, weights_path, input_source):
        super().__init__(weights_path, input_source)
        self.car_distance_threshold = 2
        self.previous_detection = {}

    def get_direction(self, detections):
        directions = {}
        for det in detections:
            x_center = (det["bbox"][0] + det["bbox"][2]) / 2
            obj_id = det["class"]

            if obj_id in self.previous_detection:
                prev_x_center = self.previous_detection[obj_id]["x_center"]

                if x_center > prev_x_center + 5:
                    direction = "Right"
                elif x_center < prev_x_center - 5:
                    direction = "Left"
                else:
                    direction = "Stationary"
            else:
                direction = "Unknown"

            directions[obj_id] = direction
            self.previous_detection[obj_id] = {"x_center": x_center}

        return directions

    def get_current_pos(self, detections):
        positions = {}
        for det in detections:
            if det["class"] == "Person":  # Controleer dat het een persoon is
                x_center = (det["bbox"][0] + det["bbox"][2]) / 2

                if x_center < 300:
                    position = "Left"
                elif x_center > 500:
                    position = "Right"
                else:
                    position = "Center"

                positions[det["class"]] = position

        return positions

    def get_safe_pos(self, detections, directions):
        safe_positions = {}
        for det in detections:
            obj_id = det["class"]
            x_center = (det["bbox"][0] + det["bbox"][2]) / 2
            direction = directions.get(obj_id, "Unknown")

            if direction == "Stationary" and (x_center < 50 or x_center > 800):
                safe_positions[obj_id] = True
            else:
                safe_positions[obj_id] = False

        return safe_positions
