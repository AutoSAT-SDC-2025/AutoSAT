import logging
import time

import numpy as np

from src.control_modes.autonomous_mode.localization.lane_detection import LaneDetector
# from multiprocessing import mp
from .line_detection.LineDetection import LineFollowingNavigation
from .localization import localization
import cv2 as cv
from .object_detection.ObjectDetection import ObjectDetection
from ...camera.camera_controller import CameraController, return_lower_rez
from ...car_variables import CarType, HunterControlMode, KartGearBox, LineDetectionDims
from ...control_modes.IControlMode import IControlMode
from ...control_modes.autonomous_mode.object_detection.TrafficDetection import TrafficManager
from ...can_interface.bus_connection import connect_to_can_interface, disconnect_from_can_interface
from ...can_interface.can_factory import select_can_controller_creator, create_can_controller
from ...util.Render import Renderer
from .obstacle_avoidance.vehicle_handler import VehicleHandler
from .obstacle_avoidance.pedestrian_handler import PedestrianHandler


# from ...misc import calculate_steering, calculate_throttle, controller_break_value, dead_man_switch, setup_listeners

# Utility function to normalize steering angle for CAN bus
def normalize_steering(angle_deg: float, max_output: float) -> float:
    # Clip the input angle to [-45, 45] for safety
    angle_deg = max(min(angle_deg, 45), -45)
    return (angle_deg / 45.0) * max_output


class AutonomousMode(IControlMode):
    """
    Main class for autonomous driving mode.
    Handles perception, decision, and actuation for the vehicle.
    """

    def __init__(self, car_type: CarType, use_checkpoint_mode=False, camera_controller=None, renderer=None,
                 data_logger_manager=None):
        # Camera setup
        self.camera_controller = camera_controller
        if self.camera_controller is None:
            try:
                self.camera_controller = CameraController()
                self.camera_controller.enable_cameras()
                self.camera_controller.setup_cameras()
            except Exception as e:
                logging.error(f"Error initializing camera controller: {e}")
                self.camera_controller = None

        self.car_type = car_type

        # CAN bus and controller setup
        self.can_bus = connect_to_can_interface(0)
        self.can_creator = select_can_controller_creator(self.car_type)
        self.can_controller = create_can_controller(self.can_creator, self.can_bus)

        # State variables for object avoidance and detection
        self.car_seen_counter = 0
        self.car_on_left = False
        self.ignore_line_detection_until = 0  # Timestamp to ignore line detection after pedestrian

        # Navigation and perception modules
        self.nav = LineFollowingNavigation(width=LineDetectionDims.WIDTH, height=LineDetectionDims.HEIGHT,
                                           mode="normal")
        self.object_detector = ObjectDetection(weights_path='assets/v5_model.pt', input_source='video')
        self.traffic_manager = TrafficManager()
        self.renderer = renderer if renderer is not None else Renderer()
        self.data_logger_manager = data_logger_manager

        # Localization
        self.localizer = localization.Localizer()
        self.lane_detector = LaneDetector()
        self.saw_ped_at = None

        # Obstacle handlers
        self.vehicle_handler = VehicleHandler(weights_path='assets/v5_model.pt', input_source='video',
                                              localizer=self.localizer, can_controller=self.can_controller,
                                              car_type=self.car_type)
        self.pedestrian_handler = PedestrianHandler(weights_path='assets/v5_model.pt', input_source='video',
                                                    can_controller=self.can_controller, car_type=self.car_type)
        self.saw_car = False
        self.saw_pedestrian = False

    def adjust_steering(self, steering_angle):
        """
        Adjusts the steering angle to the scale used by the CAN controller.
        """
        new_steering_angle = steering_angle * 576 / 90
        return max(min(new_steering_angle, 576), -576)

    def start(self):
        """
        Main loop for autonomous driving.
        Handles perception, decision making, and actuation.
        """
        logging.info("Starting autonomous mode...")
        try:
            self.can_controller.start()
            # Set initial car state based on car type
            if self.car_type == CarType.kart:
                self.can_controller.set_kart_gearbox(KartGearBox.forward)
            elif self.car_type == CarType.hunter:
                self.can_controller.set_control_mode(HunterControlMode.command_mode)

            while True:
                # Get camera images
                stitched = self.camera_controller.get_stitched_image()
                front_view = self.camera_controller.get_front_view()

                # Lane detection and localization update
                lane = self.lane_detector(return_lower_rez(front_view))
                self.localizer.update(front_view, lane)

                self.renderer.clear()

                # Line following navigation
                steering_angle, speed, visuals = self.nav.process(stitched)
                self.renderer.add_drawings_linedetection(visuals)

                # Object detection (traffic lights, cars, pedestrians)
                traffic_state, detections, draw_instructions = self.object_detector.process(front_view)
                self.renderer.add_drawings_objectdetection(draw_instructions)

                saw_red_light = traffic_state['red_light']
                speed_limit = traffic_state['speed_limit']

                # Flags for detected objects
                car_in_range = False
                pedestrian_in_range = False
                red_light_in_range = False

                # Analyze detections for relevant objects
                for det in detections:
                    distance = det.get("distance", float('inf'))
                    obj_class = det.get("class", "")
                    x1, y1, x2, y2 = det['bbox']
                    bbox_width = x2 - x1
                    bbox_height = y2 - y1
                    if obj_class == "Car" and distance <= 10 and bbox_width > 80 and bbox_height > 70:
                        car_in_range = True
                        self.saw_car = True
                    elif obj_class == "Person" and distance <= 3:
                        pedestrian_in_range = True
                        self.saw_pedestrian = True
                    if saw_red_light and distance <= 1:
                        red_light_in_range = True

                # Render visualizations
                self.renderer.render_lines(stitched)
                self.renderer.render_objects(front_view)

                # --- Decision logic for traffic and obstacles ---
                if red_light_in_range:
                    # Stop for red light
                    logging.info("Saw red light, stopping.")
                    if self.car_type == CarType.hunter:
                        self.can_controller.set_steering_and_throttle(0, 0)
                        self.can_controller.set_parking_mode(1)
                    else:
                        self.can_controller.set_throttle(0)
                        self.can_controller.set_break(100)
                elif car_in_range and self.vehicle_handler.overtake_completed is False:
                    # Overtake logic for detected car
                    logging.info("Saw car, initializing overtake")
                    self.vehicle_handler.manual_main(front_view)
                    if self.vehicle_handler.overtake_completed:
                        logging.info("Overtake completed, returning to original mode")
                elif pedestrian_in_range and self.pedestrian_handler.pedestrian_crossed() is False:
                    # Stop for pedestrian and set ignore window for line detection
                    logging.info("Saw person, stopping car")
                    self.pedestrian_handler.main(front_view)
                    if self.pedestrian_handler.pedestrian_crossed():
                        self.ignore_line_detection_until = time.time() + 7  # Ignore line detection for 7 seconds
                else:
                    # Main driving logic (normal or after pedestrian)
                    now = time.time()
                    if now < self.ignore_line_detection_until:
                        # Go straight for a few seconds after pedestrian
                        logging.info("Ignoring line detection, going straight for 5 seconds after pedestrian")
                        steering_angle = 0

                    # Log and store localization data
                    logging.info(f"Speed: {speed}, Steering: {steering_angle}")
                    logging.info(f"X: {self.localizer.x} Y: {self.localizer.y} THETA: {self.localizer.theta}")
                    self.data_logger_manager.add_location_data(self.localizer.x, self.localizer.y, self.localizer.theta)

                    # Actuate based on car type
                    if self.car_type == CarType.hunter:
                        # Hunter: negative steering, set throttle, disable parking
                        normalized_steering = -normalize_steering(steering_angle, 576)
                        self.can_controller.set_steering_and_throttle(normalized_steering, 320)
                        self.can_controller.set_parking_mode(0)
                    else:
                        # Kart: set throttle, gearbox, and steering
                        self.can_controller.set_break(0)
                        self.can_controller.set_kart_gearbox(KartGearBox.forward)
                        self.can_controller.set_throttle(30)
                        rad_steer = steering_angle * np.pi / 180
                        # rad_steer = max((min(rad_steer, -1.25)), 1.25) ##fun fact this line right here is reversed i will keep it as a reminder of how not to limit values between 2 values :)Add commentMore actions
                        rad_steer = max((min(rad_steer, 1.25)), -1.25)  ## correct way to limit a value
                        self.can_controller.set_steering(rad_steer)
        except Exception as e:
            logging.error(f"Error in autonomous mode: {e}")
        finally:
            self.stop()

    def stop(self) -> None:
        """
        Safely stop the vehicle and clean up resources.
        """
        logging.info("Stopping autonomous mode.")
        if self.car_type == CarType.hunter:
            self.can_controller.set_control_mode(HunterControlMode.idle_mode)
        else:
            self.can_controller.set_kart_gearbox(KartGearBox.neutral)
            self.can_controller.set_break(100)
        self.can_controller.stop()
        disconnect_from_can_interface(self.can_bus)