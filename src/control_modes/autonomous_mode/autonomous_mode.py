import logging
import multiprocessing as mp
from .line_detection.LineDetection import LineFollowingNavigation
from .localization import localization
from .object_detection.ObjectDetection import ObjectDetection
from ...camera.camera_controller import CameraController
from ...car_variables import CarType, HunterControlMode, KartGearBox, LineDetectionDims
from ...control_modes.IControlMode import IControlMode
from ...control_modes.autonomous_mode.object_detection.TrafficDetection import TrafficManager
from ...can_interface.bus_connection import connect_to_can_interface, disconnect_from_can_interface
from ...can_interface.can_factory import select_can_controller_creator, create_can_controller
from ...util.Render import Renderer
from .obstacle_avoidance.vehicle_handler import VehicleHandler
from .obstacle_avoidance.pedestrian_handler import PedestrianHandler

class AutonomousMode(IControlMode):

    def __init__(self, car_type: CarType, use_checkpoint_mode=False, camera_controller = None, renderer = None, data_logger_manager = None):
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
        self.can_bus = connect_to_can_interface(0)
        self.can_creator = select_can_controller_creator(self.car_type)
        self.can_controller = create_can_controller(self.can_creator, self.can_bus)

        self.car_seen_counter = 0
        self.car_on_left = False

        self.nav = LineFollowingNavigation(width=LineDetectionDims['width'], height=LineDetectionDims['height'])
        self.object_detector = ObjectDetection(weights_path='assets/v5_model.pt', input_source='video')
        self.traffic_manager = TrafficManager()
        self.renderer = renderer if renderer is not None else Renderer()

        self.data_logger_manager = data_logger_manager

        localization_manager = mp.Manager()
        self.location = localization_manager.Namespace()
        self.location.x = 0
        self.location.y = 0
        self.location.theta = 0
        self.location.img = None
        self.localization_process = mp.Process(target=localization.localization_worker, args=(self.location,))
        self.localization_process.start()

        self.vehicle_handler = VehicleHandler(weights_path='assets/v5_model.pt', input_source='video', localizer=localization_manager, can_controller=self.can_controller)
        self.pedestrian_handler = PedestrianHandler(weights_path='assets/v5_model.pt', input_source='video', can_controller=self.can_controller)

    def adjust_steering(self, steering_angle):
        new_steering_angle = steering_angle * 576 / 90
        return max(min(new_steering_angle, 576), -576)

    def start(self):
        logging.info("Starting autonomous mode...")
        try:
            self.can_controller.start()
            if self.car_type == CarType.kart:
                self.can_controller.set_kart_gearbox(KartGearBox.forward)
            elif self.car_type == CarType.hunter:
                self.can_controller.set_control_mode(HunterControlMode.command_mode)

            while True:
                stitched = self.camera_controller.get_stitched_image()
                front_view = self.camera_controller.get_front_view()

                self.location.img = front_view

                self.renderer.clear()

                steering_angle, speed, line_visuals = self.nav.process(stitched)
                self.renderer.add_drawings_objectdetection(line_visuals)

                traffic_state, detections, object_visuals = self.object_detector.process(front_view)
                self.renderer.add_drawings_objectdetection(object_visuals)

                saw_red_light = traffic_state['red_light']
                speed_limit = traffic_state['speed_limit']
                saw_car = traffic_state['car']
                saw_pedestrian = traffic_state['person']

                self.renderer.render_lines(stitched)
                self.renderer.render_objects(front_view)
                for det in detections:
                    if saw_red_light and det["distance"] <= 2:
                        logging.info("Saw red light, stopping.")
                        self.can_controller.set_steering_and_throttle(0, 0)
                        self.can_controller.set_parking_mode(1)
                    elif saw_car and det["distance"] <= 10:
                        logging.info("Saw car, initializing overtake")
                        self.vehicle_handler.main(front_view)
                        if self.vehicle_handler.goal_reached(threshold=0.1):
                            logging.info("Overtake completed, returning to original mode")
                    elif saw_pedestrian and det["distance"] <= 2:
                        logging.info("Saw person, stopping car")
                        self.pedestrian_handler.main(front_view)
                        if self.pedestrian_handler.pedestrian_crossed():
                            logging.info("Pedestrian crossed, continue driving")
                            self.pedestrian_handler.continue_driving()
                    else:
                        logging.info(f"Speed: {speed}, Steering: {steering_angle}")
                        logging.info(f"X: {self.location.x} Y: {self.location.y} THETA: {self.location.theta}")
                        self.data_logger_manager.add_location_data(self.location.x, self.location.y, self.location.theta)
                        self.can_controller.set_steering_and_throttle(-(steering_angle * 10), 320)
                        self.can_controller.set_parking_mode(0)

        except Exception as e:
            logging.error(f"Error in autonomous mode: {e}")
        finally:
            self.stop()

    def stop(self) -> None:
        logging.info("Stopping autonomous mode.")
        self.camera_controller.disable_cameras()
        self.camera_controller = None
        self.localization_process.terminate()
        self.localization_process.join()
        if self.car_type == CarType.hunter:
            self.can_controller.set_control_mode(HunterControlMode.idle_mode)
        else:
            self.can_controller.set_kart_gearbox(KartGearBox.neutral)
            self.can_controller.set_break(100)
        self.can_controller.stop()
        disconnect_from_can_interface(self.can_bus)
