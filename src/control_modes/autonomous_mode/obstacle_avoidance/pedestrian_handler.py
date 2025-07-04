from src.util.video import get_camera_config
from ..object_detection.Detection import ObjectDetection
from ....car_variables import CameraResolution, KartGearBox

class PedestrianHandler:
    """A handler for detecting and responding to pedestrian crossings.

    Attributes
    ----------
    can_controller : object
        Controller for sending steering, throttle, and brake commands.
    car_type : str
        Type of the car ('Hunter' or other).
    cams : dict
        Camera configuration.
    object_detection : ObjectDetection
        Object detection model instance.
    person_distance_threshold : int
        Distance threshold for detecting pedestrians (currently unused).
    previous_detection : dict
        Dictionary storing previous positions of pedestrians.
    initial_position : str or None
        Starting side ("Left" or "Right") of the detected pedestrian.
    current_position : str or None
        Current side of the detected pedestrian.
    direction : str or None
        Movement direction of the pedestrian.
    car_stopped : bool
        Indicates whether the car has been stopped for a pedestrian.
    initial_position_set : bool
        Indicates if the initial pedestrian position has been recorded.
    """

    def __init__(self, weights_path=None, input_source=None, can_controller=None, car_type=None):
        """Initialize the pedestrian handler with detection model and control parameters."""
        self.can_controller = can_controller
        self.car_type = car_type
        self.cams = get_camera_config()
        self.object_detection = ObjectDetection(weights_path, input_source)
        self.person_distance_threshold = 2
        self.previous_detection = {}
        self.initial_position = None
        self.current_position = None
        self.direction = None
        self.car_stopped = False
        self.initial_position_set = False

    def get_initial_position(self, detections):
        """Determine initial pedestrian position (left or right side of frame)."""
        for det in detections:
            if det["class"] == "person":
                x1, _, x2, _ = det["bbox"]
                if (x1 + x2) / 2 < CameraResolution.WIDTH / 2:
                    print("Initial position of pedestrian is on the left side of the camera")
                    self.initial_position = "Left"
                else:
                    print("Initial position of pedestrian is on the right side of the camera")
                    self.initial_position = "Right"

    def get_direction(self, detections):
        """Estimate direction of pedestrian movement (Left, Right, Stationary)."""
        for det in detections:
            if det["class"] == "person":
                x1, y1, x2, y2 = det["bbox"]
                x_center = (x1 + x2) / 2
                obj_id = "Person"

                if obj_id in self.previous_detection:
                    prev_x_center = self.previous_detection[obj_id]["x_center"]

                    if x_center > prev_x_center + 5:
                        print("Pedestrian is going right")
                        self.direction = "Right"
                    elif x_center < prev_x_center - 5:
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
        """Determine the current position of the pedestrian (Left or Right)."""
        for det in detections:
            if det["class"] == "person":
                x_center = (det["bbox"][0] + det["bbox"][2]) / 2
                print(x_center)
                if x_center < CameraResolution.WIDTH / 2:
                    print("Pedestrian is on the left side of the camera")
                    self.current_position = "Left"
                elif x_center > CameraResolution.WIDTH / 2:
                    print("Pedestrian is on the right side of the camera")
                    self.current_position = "Right"
                else:
                    self.current_position = "Unknown"
        return self.current_position

    def pedestrian_crossed(self):
        """Check whether the pedestrian has crossed the vehicle’s path."""
        if self.current_position == "Left" and self.initial_position == "Right" and self.direction == "Stationary":
            print("Pedestrian has crossed the road")
            return True
        elif self.current_position == "Right" and self.initial_position == "Left" and self.direction == "Stationary":
            print("Pedestrian has crossed the road")
            return True
        return False

    def stop_car(self, detections):
        """Stop the vehicle if a pedestrian is detected in the danger zone."""
        if self.car_stopped:
            return

        if self.car_type == 'Hunter':
            self.can_controller.set_steering_and_throttle(0, 0)
            self.can_controller.set_parking_mode(1)
        else:
            self.can_controller.set_break(100)
            self.can_controller.set_throttle(0)

        self.car_stopped = True
        print("Stopped for pedestrian")
        return True

    def main(self, front_view=None):
        """Main handler for processing pedestrian detections and vehicle response.

        Parameters
        ----------
        front_view : ndarray or similar
            The camera image/frame from the front camera.

        Returns
        -------
        bool
            True if pedestrian has crossed and car can resume, False otherwise.
        """
        detections = self.object_detection.detect_objects(front_view)
        if not self.initial_position_set:
            self.get_initial_position(detections)
            self.initial_position_set = True
        self.stop_car(detections)
        self.get_direction(detections)
        self.get_current_pos(detections)
        if self.pedestrian_crossed():
            self.car_stopped = False
            self.initial_position_set = False
            return True
        return False


if __name__ == "__main__":
    weights_path = "assets/v5_model.pt"
    input_source = "video"
    handler = PedestrianHandler(weights_path, input_source)
    handler.main()
