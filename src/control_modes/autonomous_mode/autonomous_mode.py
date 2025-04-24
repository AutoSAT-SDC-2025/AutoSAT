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
from navigation.modes.Checkpoint import Checkpoint
from stitching import Stitcher

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


def stitch_frames(frames: list) -> any:
    if not DO_STITCH:
        return frames[0]
    stitcher = Stitcher()
    result = stitcher.stitch(frames)
    return result
WIDTH = 608
HEIGHT = 320

class AutonomousMode(IControlMode, ABC):
    def __init__(self, car_type: CarType, use_checkpoint_mode=False):
        self.car_type = car_type
        self.use_checkpoint_mode = use_checkpoint_mode
        # self.checkpoint_nav = Checkpoint() if self.use_checkpoint_mode else None
        self.can_bus = connect_to_can_interface(0)

        self.can_creator = select_can_controller_creator(self.car_type)
        self.can_controller = create_can_controller(self.can_creator, self.can_bus)

        self.car_seen_counter = 0
        self.car_on_left = False
        self.nav = LineFollowingNavigation(width=WIDTH, height=HEIGHT)
        self.car_avoidance = Avoidance()
        self.object_detector = ObjectDetection(weights_path='assets/v5_model.pt', input_source='video')
        self.traffic_manager = TrafficManager()

        self.renderer = Renderer()


async def start(self):
    logging.info("Starting autonomous mode...")
    try:
        self.can_controller.start()
        if self.car_type == CarType.kart:
            self.can_controller.set_kart_gearbox(KartGearBox.forward)
        elif self.car_type == CarType.hunter:
            self.can_controller.set_control_mode(HunterControlMode.command_mode)

        if self.use_checkpoint_mode:
            logging.info("Checkpoint mode active. Delegating control to Checkpoint navigator.")
            await self.checkpoint_nav.start()
            return
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
        if not cap.isOpened():
            logging.error("Error: Could not open video file.")
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
            self.renderer.draw(stitched_frame)

            if saw_red_light:
                logging.info("Saw red light, stopping.")
                self.can_controller.set_throttle(0)
                self.can_controller.set_break(100)
            else:
                logging.info(f"Speed: {speed}, Steering: {steering_angle}")
                self.can_controller.set_steering(steering_angle)
                self.can_controller.set_throttle(speed)
                self.can_controller.set_break(0)


            # Optionally show the frame
            cv2.imshow("Stitched Output", stitched_frame)
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
