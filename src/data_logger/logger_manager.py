"""
Data logging manager for comprehensive vehicle data capture.

Provides multi-threaded logging of CAN messages, camera frames, and location data
for autonomous vehicle testing and analysis.
"""

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
    """
    Manages comprehensive data logging for vehicle testing sessions.
    
    Captures CAN messages (raw and decoded), camera frames from all views,
    and vehicle location data in organized directory structure.
    """

    def __init__(self, camera_controller=None):
        """
        Initialize data logger with CAN bus and camera connections.
        
        Args:
            camera_controller: Optional existing camera controller instance
            
        Raises:
            RuntimeError: If CAN bus connection fails
        """
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
        self.__location_log_thread = None
        
        self.log_dir = None
        self.raw_can_file = None
        self.decoded_can_file = None
        self.raw_can_writer = None
        
        self.location_file = None
        self.location_writer = None
        self.location_queue = []

    def get_can_message(self):
        """
        Receive CAN message from bus with timeout.
        
        Raises:
            RuntimeError: If CAN message reception fails
        """
        try:
            self.raw_can_message = self.can_bus.recv(0.5)
        except Exception as e:
            raise RuntimeError(f"Error receiving CAN message: {e}")

    def enable_logger(self):
        """
        Start data logging with all threads and create log directory structure.
        
        Creates timestamped session directory and starts concurrent logging
        for CAN data, camera frames, and location data.
        """
        self.enabled = True
        self.__camera_log_thread = threading.Thread(target=self.__log_camera_frames, daemon=True)
        self.__can_log_thread = threading.Thread(target=self.__log_can_data, daemon=True)
        self.__location_log_thread = threading.Thread(target=self.__log_location_data, daemon=True)
        
        self.create_folder_structure()
        
        self.__camera_log_thread.start()
        self.__can_log_thread.start()
        self.__location_log_thread.start()
        
        logging.info("Enabled logger")

    def disable_logger(self):
        """
        Stop all logging threads and close files safely.
        
        Gracefully shuts down logging operations and ensures all data
        is written to disk before cleanup.
        """
        self.enabled = False

        if self.__can_log_thread and self.__can_log_thread.is_alive():
            self.__can_log_thread.join(timeout=2.0)

        if self.__camera_log_thread and self.__camera_log_thread.is_alive():
            self.__camera_log_thread.join(timeout=2.0)
            
        if self.__location_log_thread and self.__location_log_thread.is_alive():
            self.__location_log_thread.join(timeout=2.0)

        if self.raw_can_file:
            self.raw_can_file.close()
            self.raw_can_file = None

        if self.decoded_can_file:
            self.decoded_can_file.close()
            self.decoded_can_file = None
            
        if self.location_file:
            self.location_file.close()
            self.location_file = None
            
        logging.info("Disabled logger")

    def create_folder_structure(self):
        """
        Create organized directory structure for logging session.
        
        Creates timestamped session folder with subdirectories for CAN data,
        images (all camera views), and location data files.
        """
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
        
        self.location_file = open(os.path.join(self.log_dir, "location_data.csv"), 'w', newline='')
        self.location_writer = csv.writer(self.location_file)
        self.location_writer.writerow(["timestamp", "x", "y", "theta"])

        logging.info(f"Created log directory: {self.log_dir}")
        
    def add_location_data(self, x, y, theta):
        """
        Queue location data for logging.
        
        Args:
            x: X coordinate position
            y: Y coordinate position  
            theta: Vehicle heading angle
        """
        if not self.enabled:
            return

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")[:-3]
        self.location_queue.append((timestamp, x, y, theta))

    def __log_can_data(self):
        """
        Internal thread function for continuous CAN data logging.
        
        Logs both raw CAN messages (CSV) and decoded messages (JSON)
        with timestamps for correlation with other data streams.
        """
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
        """
        Internal thread function for continuous camera frame logging.
        
        Captures and saves timestamped images from all camera views
        (front, left, right, stitched panoramic, top-down).
        """
        while self.enabled:
            camera_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")[:-3]
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
                
    def __log_location_data(self):
        """
        Internal thread function for batched location data logging.
        
        Processes queued location entries and writes them to CSV
        with timestamp, position, and heading information.
        """
        while self.enabled:
            try:
                location_batch = self.location_queue.copy()
                self.location_queue = []
                    
                if location_batch and self.location_writer:
                    for entry in location_batch:
                        self.location_writer.writerow(entry)
                    self.location_file.flush()
            except Exception as e:
                logging.error(f"Error logging location data: {e}")