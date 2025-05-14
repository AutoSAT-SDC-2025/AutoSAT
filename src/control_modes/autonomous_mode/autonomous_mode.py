from abc import ABC

from src.util.video import get_camera_config, validate_camera_config
from src.multi_camera_calibration import CalibrationData, RenderDistance
from .line_detection.LineDetection import LineFollowingNavigation
from .object_detection.ObjectDetection import ObjectDetection
from .avoidance.Avoidance import Avoidance
from ...car_variables import CarType, HunterControlMode, KartGearBox
from ...control_modes.IControlMode import IControlMode
from ...control_modes.autonomous_mode.object_detection.TrafficDetection import TrafficManager
from ...can_interface.bus_connection import connect_to_can_interface, disconnect_from_can_interface
from ...can_interface.can_factory import select_can_controller_creator, create_can_controller
from ...util.Render import Renderer
# from navigation.modes.Checkpoint import Checkpoint
# from stitching import Stitcher
from .localization import localization
import multiprocessing as mp

import cv2
import logging

WIDTH = 848
HEIGHT = 480

LineDetectionDims = {
    'width': 720,
    'height': 720
}

class AutonomousMode(IControlMode, ABC):

    def __init__(self, car_type: CarType, use_checkpoint_mode=False):

        self.data = CalibrationData(
            path="assets/calibration/latest.npz",
            input_shape=(1920, 1080),
            output_shape=(LineDetectionDims['width'], LineDetectionDims['height']),
            render_distance=RenderDistance(
                front=12.0,
                sides=6.0
            )
        )

        self.captures = None
        self.car_type = car_type
        self.can_bus = connect_to_can_interface(0)

        self.can_creator = select_can_controller_creator(self.car_type)
        self.can_controller = create_can_controller(self.can_creator, self.can_bus)
        self.cams = get_camera_config()

        self.car_seen_counter = 0
        self.car_on_left = False
        self.nav = LineFollowingNavigation(width=LineDetectionDims['width'], height=LineDetectionDims['height'])
        self.object_detector = ObjectDetection(weights_path='assets/v5_model.pt', input_source='video')
        self.traffic_manager = TrafficManager()

        self.renderer = Renderer()

        localization_manager = mp.Manager()
        self.location = localization_manager.Namespace()
        self.location.x = 0 
        self.location.y = 0 
        self.location.theta = 0 
        self.location.img = None
        self.localization_process = mp.Process(target=localization.localization_worker, args=(self.location,))
        self.localization_process.start()

    def setup_cameras(self):
        self.captures = {}
        for cam_name in ['left', 'front', 'right']:
            cam_path = self.cams[cam_name]
            cap = cv2.VideoCapture(cam_path)

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

            if not cap.isOpened():
                raise RuntimeError(f"Failed to open {cam_name} camera at {cam_path}")

            self.captures[cam_name] = cap

    def capture(self):
        frames = {}
        for cam_name, cap in self.captures.items():
            for _ in range(2):  # Warm-up frames
                cap.read()

            ret, frame = cap.read()
            if not ret:
                raise RuntimeError(f"Failed to read frame from {cam_name} camera.")

            frames[cam_name] = frame

        try:
            top_down = self.data.transform([frames['left'], frames['front'], frames['right']])
        except Exception as e:
            raise RuntimeError(f"Stitching error: {e}")

        front_frame = frames['front']
        self.location.img = front_frame
        front_frame = cv2.resize(front_frame, (WIDTH, HEIGHT))

        return top_down, front_frame

    def adjust_steering(self, steering_angle):
        new_steering_angle = steering_angle * 576 / 90
        if new_steering_angle > 576:
            new_steering_angle = 576
        elif new_steering_angle < -576:
            new_steering_angle = -576
        return new_steering_angle

    def start(self):
        logging.info("Starting autonomous mode...")
        try:
            self.can_controller.start()
            if self.car_type == CarType.kart:
                self.can_controller.set_kart_gearbox(KartGearBox.forward)
            elif self.car_type == CarType.hunter:
                self.can_controller.set_control_mode(HunterControlMode.command_mode)

            if not validate_camera_config(self.cams):
                raise RuntimeError("No cameras connected or invalid camera configuration")
            # Take full-resolution, then ->
            # use stiched for line detection
            # use downscaled of front-camera for object detection

            while True:
                # frames = []
                # for cap in caps:
                #     ret, frame = cap.read()
                #     if not ret:
                #         continue
                #     frames.append(frame)
                #
                # if not frames:
                #     continue

                top_view, front_view = self.capture()

                # Clear previous drawings before new frame
                self.renderer.clear()

                # 576 and -576 are steering values. we get angles.

                steering_angle, speed, line_visuals = self.nav.process(top_view)
                self.renderer.add_drawings(line_visuals)
                # === Object detection & traffic ===
                traffic_state, detections, object_visuals = self.object_detector.process(front_view)
                self.renderer.add_drawings(object_visuals)

                saw_red_light = traffic_state['red_light']
                speed_limit = traffic_state['speed_limit']

                # avoidance_steering, speed_scale, avoidance_drawings = self.car_avoidance.process(stitched_frame, detections)
                # self.renderer.add_drawings(avoidance_drawings)

                # steering_angle += avoidance_steering
                # speed *= speed_scale

                # Draw everything
                self.renderer.render(top_view)
                #
                if saw_red_light:
                    logging.info("Saw red light, stopping.")
                    self.can_controller.set_steering_and_throttle(0, 0)
                    self.can_controller.set_parking_mode(1)
                else:
                    logging.info(f"Speed: {speed}, Steering: {steering_angle}")
                    logging.info(f"X: {self.location.x} Y: {self.location.y} THETA: {self.location.theta}")
                    self.can_controller.set_steering_and_throttle(-(steering_angle * 10), 320)
                    self.can_controller.set_parking_mode(0)

                # Optionally show the frame
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except Exception as e:
            logging.error(f"Error in autonomous mode: {e}")
        finally:
            self.stop()
            cv2.destroyAllWindows()

    def stop(self) -> None:
        logging.info("Stopping autonomous mode.")
        self.localization_process.terminate()
        self.localization_process.join()
        for cap in getattr(self, 'captures', {}).values():
            cap.release()
        if self.car_type == CarType.hunter:
            self.can_controller.set_control_mode(HunterControlMode.idle_mode)
        else:
            self.can_controller.set_kart_gearbox(KartGearBox.neutral)
            self.can_controller.set_break(100)
        cv2.destroyAllWindows()
        disconnect_from_can_interface(self.can_bus)
