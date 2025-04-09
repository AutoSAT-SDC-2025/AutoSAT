from .line_detection.LineDetection import LineFollowingNavigation
from .object_detection.ObjectDetection import ObjectDetection
from ...car_variables import CarType, HunterControlMode, KartGearBox
from ...control_modes.IControlMode import IControlMode
from ...control_modes.autonomous_mode.object_detection.TrafficDetection import TrafficManager
from ...can_interface.bus_connection import connect_to_can_interface, disconnect_from_can_interface
from ...can_interface.can_factory import select_can_controller_creator, create_can_controller
from stitching import Stitcher

import cv2
import logging

def get_connected_cameras(max_devices=5):
    cameras = []
    for idx in range(max_devices):
        cap = cv2.VideoCapture(idx)
        if cap.read()[0]:
            cameras.append(idx)
        cap.release()
    return cameras

def stitch_frames(frames: list) -> any:
    stitcher = Stitcher()
    result = stitcher.stitch(frames)
    return result

class AutonomousMode(IControlMode):
    def __init__(self, car_type: CarType):
        self.car_type = car_type
        self.can_bus = connect_to_can_interface(0)
        can_creator = select_can_controller_creator(self.car_type)
        self.can_controller = create_can_controller(can_creator, self.can_bus)

        self.nav = LineFollowingNavigation(width=848, height=480)
        self.object_detector = ObjectDetection(weights_path='../../../assets/v5_model.pt', input_source='video')
        self.traffic_manager = TrafficManager()

    async def start(self):
        logging.info("Starting autonomous mode...")
        try:
            self.can_controller.start()
            if self.car_type == CarType.kart:
                self.can_controller.set_kart_gearbox(KartGearBox.forward)
            elif self.car_type == CarType.hunter:
                self.can_controller.set_control_mode(HunterControlMode.command_mode)

            cap = cv2.VideoCapture('../../../assets/default.mp4')
            if not cap.isOpened():
                logging.error("Error: Could not open video file.")
                return

            camera_ids = get_connected_cameras()
            caps = [cv2.VideoCapture(cam_id) for cam_id in camera_ids]

            while True:
                frames = []
                for cap in caps:
                    ret, frame = cap.read()
                    if not ret:
                        continue
                    frames.append(frame)

                if not frames:
                    continue

                stitched_frame = len(frames) == 1 and frames[0] or stitch_frames(frames)

                steering_angle, speed, viz_img, end_x = self.nav.process(stitched_frame)
                traffic_state, detections = self.object_detector.process(stitched_frame)

                saw_red_light = traffic_state['red_light']
                speed_limit = traffic_state['speed_limit']

                if saw_red_light:
                    speed = 0
                elif speed_limit:
                    speed = min(speed, speed_limit)

                self.can_controller.set_steering(steering_angle)
                self.can_controller.set_throttle(speed)
                self.can_controller.set_break(0)

                cv2.imshow("Line Following", viz_img)
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
