from abc import ABC

from .line_detection.LineDetection import LineFollowingNavigation
from .object_detection.ObjectDetection import ObjectDetection
from .avoidance.Avoidance import Avoidance
from ...car_variables import CarType, HunterControlMode, KartGearBox
from ...control_modes.IControlMode import IControlMode
from ...control_modes.autonomous_mode.object_detection.TrafficDetection import TrafficManager
from ...can_interface.bus_connection import connect_to_can_interface, disconnect_from_can_interface
from ...can_interface.can_factory import select_can_controller_creator, create_can_controller
from ...util.Render import Renderer
#from navigation.modes.Checkpoint import Checkpoint
#from stitching import Stitcher

import cv2
import logging

DO_STITCH = False


def get_connected_cameras(max_devices=5):
    cameras = []
    for idx in range(max_devices):
        cap = cv2.VideoCapture(idx)
        if cap.read()[0]:
            cameras.append(idx)
        cap.release()
    return cameras


WIDTH = 848
HEIGHT = 480

class AutonomousMode(IControlMode, ABC):
    def __init__(self, car_type: CarType, use_checkpoint_mode=False):
        self.car_type = car_type
        self.can_bus = connect_to_can_interface(0)

        self.can_creator = select_can_controller_creator(self.car_type)
        self.can_controller = create_can_controller(self.can_creator, self.can_bus)

        self.car_seen_counter = 0
        self.car_on_left = False
        self.nav = LineFollowingNavigation(width=WIDTH, height=HEIGHT)
        self.object_detector = ObjectDetection(weights_path='assets/v5_model.pt', input_source='video')
        self.traffic_manager = TrafficManager()

        self.renderer = Renderer()


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


            cap = cv2.VideoCapture(0)

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
            if not cap.isOpened():
                logging.error("Error: Could not open video channel.")
                return

            # camera_ids = get_connected_cameras()
            # caps = [cv2.VideoCapture(cam_id) for cam_id in camera_ids]

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

                ret, stitched_frame = cap.read()

                # Clear previous drawings before new frame
                self.renderer.clear()

                #576 and -576 are steering values. we get angles.


                steering_angle, speed, line_visuals = self.nav.process(stitched_frame)
                self.renderer.add_drawings(line_visuals)
                # === Object detection & traffic ===
                traffic_state, detections, object_visuals = self.object_detector.process(stitched_frame)
                self.renderer.add_drawings(object_visuals)

                saw_red_light = traffic_state['red_light']
                speed_limit = traffic_state['speed_limit']

               # avoidance_steering, speed_scale, avoidance_drawings = self.car_avoidance.process(stitched_frame, detections)
               # self.renderer.add_drawings(avoidance_drawings)

               # steering_angle += avoidance_steering
               # speed *= speed_scale

                # Draw everything
                self.renderer.render(stitched_frame)
                #
                if saw_red_light:
                    logging.info("Saw red light, stopping.")
                    self.can_controller.set_steering_and_throttle(0, 0)
                    self.can_controller.set_parking_mode(1)
                else:
                    logging.info(f"Speed: {speed}, Steering: {steering_angle}")
                    self.can_controller.set_steering_and_throttle(-(steering_angle * 19), 200)
                    self.can_controller.set_parking_mode(0)

                # Optionally show the frame
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except Exception as e:
            logging.error(f"Error in autonomous mode: {e}")
        finally:
            self.stop()
            cap.release()
            cv2.destroyAllWindows()


    def stop(self) -> None:
        logging.info("Stopping autonomous mode.")
        if self.car_type == CarType.hunter:
            self.can_controller.set_control_mode(HunterControlMode.idle_mode)
        else:
            self.can_controller.set_kart_gearbox(KartGearBox.neutral)
            self.can_controller.set_break(100)
        cv2.destroyAllWindows()
        disconnect_from_can_interface(self.can_bus)
