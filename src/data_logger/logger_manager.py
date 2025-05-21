import csv
import json
import logging
import os
import threading
from datetime import datetime

import cv2

from src.can_interface import bus_connection
from src.camera.camera_controller import CameraController
from src.can_interface.can_decoder import CanDecoder

class DataLoggerManager:

    def __init__(self, camera_controller=None):
        try:
            self.can_bus = bus_connection.connect_to_can_interface(0)
        except Exception as e:
            raise RuntimeError(f"Error connecting to CAN bus: {e}")
        self.camera_controller = camera_controller
        if self.camera_controller is None:
            try:
                self.camera_controller = CameraController()
                self.camera_controller.enable_cameras()
                self.camera_controller.setup_cameras()
            except Exception as e:
                logging.error(f"Error initializing camera controller: {e}")
                self.camera_controller = None
        self.can_decoder = CanDecoder()
        self.raw_can_message = None
        self.decoded_can_message = None
        self.enabled = False

        self.__camera_log_thread = None
        self.__can_log_thread = None

        self.log_dir = None
        self.raw_can_file = None
        self.decoded_can_file = None
        self.raw_can_writer = None

    def get_can_message(self):
        try:
            self.raw_can_message = self.can_bus.recv(0.5)
        except Exception as e:
            raise RuntimeError(f"Error recieving CAN message: {e}")

    def enable_logger(self):
        self.enabled = True
        self.__camera_log_thread = threading.Thread(target=self.__log_camera_frames, daemon=True)
        self.__can_log_thread = threading.Thread(target=self.__log_can_data, daemon=True)
        self.__camera_log_thread.start()
        self.__can_log_thread.start()
        self.create_folder_structure()
        logging.info("Enabled logger")

    def disable_logger(self):
        self.enabled = False

        if self.__can_log_thread and self.__can_log_thread.is_alive():
            self.__can_log_thread.join(timeout=2.0)

        if self.__camera_log_thread and self.__camera_log_thread.is_alive():
            self.__camera_log_thread.join(timeout=2.0)

        if self.raw_can_file:
            self.raw_can_file.close()
            self.raw_can_file = None

        if self.decoded_can_file:
            self.decoded_can_file.close()
            self.decoded_can_file = None
        logging.info("Disabled logger")

    def create_folder_structure(self):
        """Create folder structure for storing logs"""
        session_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")[:-3]
        base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "logs")
        self.log_dir = os.path.join(base_dir, f"session_{session_timestamp}")

        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(os.path.join(self.log_dir, "can"), exist_ok=True)
        os.makedirs(os.path.join(self.log_dir, "images"), exist_ok=True)

        for view in ["front", "left", "right", "stitched", "topdown"]:
            os.makedirs(os.path.join(self.log_dir, "images", view), exist_ok=True)

        self.raw_can_file = open(os.path.join(self.log_dir, "can", "raw_messages.csv"), 'w', newline='')
        self.raw_can_writer = csv.writer(self.raw_can_file)
        self.raw_can_writer.writerow(["timestamp", "id", "data_hex"])

        self.decoded_can_file = open(os.path.join(self.log_dir, "can", "decoded_messages.json"), 'w')

        logging.info(f"Created log directory: {self.log_dir}")

    def __log_can_data(self):
        while self.enabled:
            try:
                can_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")[:-3]
                self.get_can_message()

                hex_data = ' '.join([f"{b:02X}" for b in self.raw_can_message.data])
                self.raw_can_writer.writerow([can_timestamp, f"0x{self.raw_can_message.arbitration_id:X}", hex_data])
                self.raw_can_file.flush()


                decoded = self.can_decoder.decode_message(self.raw_can_message)
                json_obj = decoded.to_dict()
                json_obj["timestamp"] = can_timestamp

                self.decoded_can_file.write(json.dumps(json_obj) + "\n")
                self.decoded_can_file.flush()

            except Exception as e:
                logging.error(f"Error logging CAN data: {e}")

    def __log_camera_frames(self):
        while self.enabled:
            camera_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")[:-3]
            # self.camera_controller.capture_camera_frames()

            try:
                front_frame = self.camera_controller.get_front_view()
                if front_frame is not None:
                    cv2.imwrite(os.path.join(self.log_dir, "images", "front", f"frame_{camera_timestamp}.jpg"),
                                front_frame)
            except Exception as e:
                logging.error(f"Error saving front view: {e}")

            try:
                left_frame = self.camera_controller.get_left_view()
                if left_frame is not None:
                    cv2.imwrite(os.path.join(self.log_dir, "images", "left", f"frame_{camera_timestamp}.jpg"),
                                left_frame)
            except Exception as e:
                logging.error(f"Error saving left view: {e}")

            try:
                right_frame = self.camera_controller.get_right_view()
                if right_frame is not None:
                    cv2.imwrite(os.path.join(self.log_dir, "images", "right", f"frame_{camera_timestamp}.jpg"), right_frame)
            except Exception as e:
                logging.error(f"Error saving right view: {e}")

            try:
                stitched_frame = self.camera_controller.get_stitched_image()
                if stitched_frame is not None:
                    cv2.imwrite(os.path.join(self.log_dir, "images", "stitched", f"frame_{camera_timestamp}.jpg"), stitched_frame)
            except Exception as e:
                logging.error(f"Error saving stitched view: {e}")

            try:
                topdown_frame = self.camera_controller.get_top_down_view()
                if topdown_frame is not None:
                    cv2.imwrite(os.path.join(self.log_dir, "images", "topdown", f"frame_{camera_timestamp}.jpg"), topdown_frame)
            except Exception as e:
                logging.error(f"Error saving top down view: {e}")